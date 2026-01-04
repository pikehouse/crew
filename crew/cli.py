"""CLI and REPL interface for crew."""

from __future__ import annotations

import asyncio
import signal
import subprocess
import sys
import threading
import queue
from pathlib import Path
from dataclasses import dataclass
from typing import Literal

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from rich.console import Console

from crew.state import load_state, save_state, ensure_crew_dir
from crew.pid import check_and_acquire_lock, remove_pid_file
from crew.display import (
    print_banner,
    print_help,
    print_error,
    print_info,
    print_warning,
    print_agent_created,
    print_agent_step,
    print_agent_done,
    print_agent_merged,
    print_peek,
    print_git_status_panels,
    get_status_icon,
)
from crew.runner import spawn_agent, spawn_worker, step_agent, cleanup_agent, assign_task, complete_task, get_ready_tasks, shutdown_agent
from crew.crew_logging import read_log_tail, read_all_logs
from crew.git import get_worktree_list, has_uncommitted_changes, remove_worktree

console = Console()


@dataclass
class RunnerEvent:
    """Event from background runner."""
    type: Literal["step", "done", "merged", "error", "stopped", "assigned"]
    agent_name: str
    message: str = ""


class BackgroundRunner:
    """Runs agents in background thread, polling for work."""

    def __init__(self, state, project_root: Path):
        self.state = state
        self.project_root = project_root
        self.running = False
        self.thread: threading.Thread | None = None
        self.events: queue.Queue = queue.Queue()
        # Track which agents are currently being stepped (to avoid double-stepping)
        self._stepping: set = set()
        # Track recently completed tasks to prevent immediate re-assignment
        self._recently_completed: set = set()
        # Lock for thread-safe operations
        self._lock = threading.Lock()
        # Lock to serialize complete_task calls (prevents git checkout race conditions)
        self._merge_lock = threading.Lock()
        # Store executor reference for clean shutdown
        self._executor = None

    def start(self):
        """Start the background runner."""
        if self.running:
            return False
        self.running = True
        self._stepping = set()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        """Stop the background runner gracefully."""
        self.running = False

        # Shutdown the executor without waiting for pending tasks
        if self._executor:
            try:
                self._executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                # Python < 3.9 doesn't have cancel_futures
                self._executor.shutdown(wait=False)
            self._executor = None

        if self.thread:
            self.thread.join(timeout=2)
            self.thread = None

    def _get_idle_agents(self) -> list:
        """Get agents that are idle and ready for work."""
        return [a for a in self.state.agents.values() if a.status == "idle"]

    def _get_working_agents(self) -> list:
        """Get agents that are ready or working and not currently being stepped."""
        with self._lock:
            return [a for a in self.state.agents.values()
                    if a.status in ("ready", "working") and a.name not in self._stepping]

    def _get_assigned_tasks(self) -> set:
        """Get set of task IDs already assigned to agents."""
        return {a.task for a in self.state.agents.values() if a.task}

    def _assign_work(self):
        """Poll for ready tasks and assign to idle agents."""
        idle_agents = self._get_idle_agents()
        if not idle_agents:
            return

        # Get ready tasks that aren't already assigned or recently completed
        assigned = self._get_assigned_tasks()
        with self._lock:
            skip_tasks = assigned | self._recently_completed
        ready_tasks = [t for t in get_ready_tasks() if t not in skip_tasks]

        # Assign tasks to idle agents
        for agent, task_id in zip(idle_agents, ready_tasks):
            try:
                assign_task(agent, task_id, self.state, self.project_root)
                self.events.put(RunnerEvent("assigned", agent.name, task_id))
            except Exception as e:
                self.events.put(RunnerEvent("error", agent.name, f"Failed to assign {task_id}: {e}"))

    def _step_one_agent(self, agent):
        """Step a single agent (runs in thread pool)."""
        from crew.runner import TEST_FAILURE_PROMPT

        try:
            output = step_agent(agent, self.state, project_root=self.project_root)
            preview = output[:60].replace('\n', ' ') if output else ""
            self.events.put(RunnerEvent("step", agent.name, preview))

            # Loop while agent keeps saying DONE (for test failure retries)
            while agent.is_done:
                task_id = agent.task
                branch = agent.branch
                # Complete the task (merge, cleanup, return to idle)
                # Use merge lock to serialize git checkout operations
                try:
                    with self._merge_lock:
                        success, test_output = complete_task(agent, self.state, self.project_root)
                    if success:
                        # Track as recently completed to prevent re-assignment race
                        if task_id:
                            with self._lock:
                                self._recently_completed.add(task_id)
                        self.events.put(RunnerEvent("done", agent.name, task_id or ""))
                        self.events.put(RunnerEvent("merged", agent.name, branch))
                        break  # Successfully completed
                    else:
                        # Tests failed - step agent with test output, then loop to check again
                        self.events.put(RunnerEvent("error", agent.name, f"Tests failed, agent will retry"))
                        step_agent(
                            agent, self.state,
                            prompt=TEST_FAILURE_PROMPT.format(test_output=test_output),
                            project_root=self.project_root
                        )
                        # Loop continues - if agent says DONE again, we'll try complete_task again
                except Exception as e:
                    self.events.put(RunnerEvent("error", agent.name, f"Complete failed: {e}"))
                    break

        except Exception as e:
            self.events.put(RunnerEvent("error", agent.name, str(e)))
        finally:
            with self._lock:
                self._stepping.discard(agent.name)

    def _run_loop(self):
        """Main runner loop - assigns work and steps agents in parallel."""
        import time
        from concurrent.futures import ThreadPoolExecutor

        self._executor = ThreadPoolExecutor(max_workers=10)
        try:
            while self.running:
                # First, try to assign work to idle agents
                self._assign_work()

                # Get agents that need stepping (and aren't already being stepped)
                to_step = self._get_working_agents()

                if to_step:
                    # Mark them as being stepped
                    with self._lock:
                        for agent in to_step:
                            self._stepping.add(agent.name)

                    # Submit all to thread pool (non-blocking)
                    for agent in to_step:
                        if self.running:  # Check before submitting
                            self._executor.submit(self._step_one_agent, agent)

                # Check if anyone is still working or idle
                all_agents = list(self.state.agents.values())
                working_or_stepping = [a for a in all_agents
                                       if a.status in ("ready", "working") or a.name in self._stepping]
                idle = [a for a in all_agents if a.status == "idle"]

                if not working_or_stepping and not idle:
                    # Everyone is done or stuck
                    self.running = False
                    self.events.put(RunnerEvent("stopped", "", "All work completed"))
                    break

                # Brief pause before next poll
                time.sleep(1)
        finally:
            # Clean up the executor
            if self._executor:
                try:
                    self._executor.shutdown(wait=False, cancel_futures=True)
                except TypeError:
                    # Python < 3.9 doesn't have cancel_futures
                    self._executor.shutdown(wait=False)
                self._executor = None

    def drain_events(self) -> list[RunnerEvent]:
        """Get all pending events."""
        events = []
        while True:
            try:
                events.append(self.events.get_nowait())
            except queue.Empty:
                break
        return events

    @property
    def is_running(self) -> bool:
        return self.running


# Global runner instance
_runner: BackgroundRunner | None = None

# Global read-only mode flag
_read_only_mode: bool = False

# Global state and project root for signal handlers
_state = None
_project_root: Path | None = None

# Flag to track if we're already shutting down (prevent re-entry)
_shutting_down: bool = False

# Commands that modify state and are blocked in read-only mode
_MODIFYING_COMMANDS = frozenset({
    "spawn", "run", "work", "stop", "kill", "cleanup", "merge",
    "reset", "new", "dep", "assign"
})


def graceful_shutdown(signum: int | None = None, frame=None) -> None:
    """Handle graceful shutdown on SIGINT/SIGTERM.

    Stops the runner, shuts down working agents, saves state,
    cleans up PID file, and exits without traceback.
    """
    global _runner, _state, _project_root, _read_only_mode, _shutting_down

    # Prevent re-entry if already shutting down
    if _shutting_down:
        return
    _shutting_down = True

    # Suppress KeyboardInterrupt traceback
    try:
        # Stop the runner if running
        if _runner and _runner.is_running:
            _runner.stop()

        # Shutdown working agents and save state
        if _state and _project_root:
            for agent in list(_state.agents.values()):
                if agent.status in ("ready", "working"):
                    try:
                        shutdown_agent(agent, _state, _project_root)
                    except Exception:
                        pass  # Best effort on shutdown

            # Save state
            try:
                save_state(_state, _project_root)
            except Exception:
                pass  # Best effort

        # Clean up PID file
        if not _read_only_mode and _project_root:
            try:
                remove_pid_file(_project_root)
            except Exception:
                pass  # Best effort

        # Print a clean exit message
        console.print("\n[dim]interrupted, bye[/dim]")

    except Exception:
        pass  # Suppress any errors during shutdown

    # Exit cleanly without traceback
    sys.exit(0)


def get_prompt(state) -> str:
    """Generate the prompt string."""
    global _runner, _read_only_mode

    workers = len(state.agents)
    working = len([a for a in state.agents.values() if a.status in ("ready", "working")])
    idle = len([a for a in state.agents.values() if a.status == "idle"])

    # Build the read-only prefix if applicable
    ro_prefix = "[read-only] " if _read_only_mode else ""

    if _runner and _runner.is_running:
        if working:
            return f"{ro_prefix}[{working}/{workers} working] crew> "
        elif idle:
            return f"{ro_prefix}[{idle}/{workers} waiting] crew> "
        else:
            return f"{ro_prefix}[{workers} agents] crew> "
    elif workers:
        return f"{ro_prefix}[{workers} workers] crew> "
    return f"{ro_prefix}crew> "


def run_tk(*args: str) -> str:
    """Run a tk (ticket) command and return output."""
    try:
        result = subprocess.run(
            ["tk", *args],
            capture_output=True,
            text=True,
        )
        return result.stdout + result.stderr
    except FileNotFoundError:
        return "Error: tk command not found. Is ticket installed?"


def _next_worker_name(state) -> str:
    """Get next available single-letter worker name (a, b, c, ...)."""
    existing = set(state.agents.keys())
    for i in range(26):
        name = chr(ord('a') + i)
        if name not in existing:
            return name
    # Fall back to a1, a2, etc if we run out of letters
    for i in range(100):
        name = f"a{i}"
        if name not in existing:
            return name
    return f"worker{len(existing)}"


def cmd_spawn(state, args: list[str], project_root: Path) -> None:
    """Spawn worker agent(s).

    Usage:
        spawn <name>       - spawn a single named worker
        spawn <n>          - spawn n workers with auto names (a, b, c, ...)
        spawn <name> <id>  - spawn named worker with task
    """
    if not args:
        print_error("Usage: spawn <name|count> [task-id]")
        return

    first_arg = args[0]

    # Check if first arg is a number (spawn N workers)
    if first_arg.isdigit():
        count = int(first_arg)
        if count < 1 or count > 26:
            print_error("Can spawn 1-26 workers at a time")
            return

        created = []
        for _ in range(count):
            name = _next_worker_name(state)
            try:
                spawn_worker(name, state, project_root)
                created.append(name)
            except Exception as e:
                print_error(f"Failed to spawn {name}: {e}")
                break

        if created:
            console.print(f"[green]✓[/green] Created {len(created)} workers: [bold]{', '.join(created)}[/bold]")
            print_info("Use 'run' to auto-assign work")
        return

    # Single named worker
    name = first_arg
    task_id = args[1] if len(args) > 1 else None

    # Check if agent already exists
    if state.get_agent(name):
        print_error(f"Agent '{name}' already exists")
        return

    try:
        # Create idle worker
        agent = spawn_worker(name, state, project_root)
        console.print(f"[green]✓[/green] Created worker [bold]{name}[/bold]")

        # If task specified, assign it immediately
        if task_id:
            try:
                assign_task(agent, task_id, state, project_root)
                console.print(f"  [cyan]▶[/cyan] Assigned {task_id}")
            except Exception as e:
                print_error(f"Failed to assign task: {e}")
        else:
            print_info(f"Worker idle. Use 'run' to auto-assign work, or 'assign {name} <task-id>'")

    except Exception as e:
        print_error(f"Failed to spawn worker: {e}")


def cmd_run(state, args: list[str], project_root: Path) -> None:
    """Start running all agents in background."""
    global _runner

    if _runner and _runner.is_running:
        print_info("Already running. Use 'stop' to pause.")
        return

    if not state.active_agents:
        print_info("No active agents to run")
        return

    _runner = BackgroundRunner(state, project_root)
    _runner.start()
    print_info(f"Started {len(state.active_agents)} agent(s) in background. Use 'stop' to pause.")


def cmd_stop(state, args: list[str], project_root: Path) -> None:
    """Stop the background runner and shutdown working agents."""
    global _runner

    if not _runner or not _runner.is_running:
        print_info("Not running")
        return

    _runner.stop()

    # Shutdown each working agent and track results
    paused_count = 0
    completed_count = 0

    for agent in list(state.agents.values()):
        if agent.status in ("ready", "working"):
            status = shutdown_agent(agent, state, project_root)
            if status == "done":
                completed_count += 1
            elif status == "partial":
                paused_count += 1
            # "nothing" status doesn't count as paused or completed

    print_info(f"Stopped. {paused_count} agents paused mid-work, {completed_count} completed.")


def cmd_reset(state, args: list[str], project_root: Path) -> None:
    """Reset everything to a clean state at a specific git commit.

    Usage: reset <commit-hash>

    This will:
    - Stop the runner
    - Remove all agent worktrees
    - Delete all agent branches
    - Clear crew state
    - Git reset --hard to the commit
    - Git clean -fd to remove untracked files
    """
    global _runner

    if not args:
        print_error("Usage: reset <commit-hash>")
        print_info("Example: reset HEAD~5   (go back 5 commits)")
        print_info("Example: reset abc123   (reset to specific commit)")
        return

    commit = args[0]

    # Confirm this destructive action
    console.print(f"[bold red]WARNING:[/bold red] This will:")
    console.print(f"  • Stop all agents")
    console.print(f"  • Remove all worktrees in agents/")
    console.print(f"  • Delete all agent/* branches")
    console.print(f"  • Clear .crew/state.json")
    console.print(f"  • git reset --hard {commit}")
    console.print(f"  • git clean -fd (preserves .tickets)")
    console.print()

    try:
        confirm = console.input("[bold]Type 'yes' to confirm: [/bold]")
    except (KeyboardInterrupt, EOFError):
        print_info("Cancelled")
        return

    if confirm.strip().lower() != "yes":
        print_info("Cancelled")
        return

    console.print()

    # Stop runner if running
    if _runner and _runner.is_running:
        _runner.stop()
        console.print("[dim]Stopped runner[/dim]")

    # Remove all worktrees
    from crew.git import remove_worktree, run_git, GitError

    agents_dir = project_root / "agents"
    if agents_dir.exists():
        for worktree_dir in agents_dir.iterdir():
            if worktree_dir.is_dir():
                try:
                    remove_worktree(worktree_dir)
                    console.print(f"[dim]Removed worktree {worktree_dir.name}[/dim]")
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not remove {worktree_dir}: {e}[/yellow]")

    # Prune worktrees (clean up any stale references)
    try:
        run_git("worktree", "prune", cwd=project_root)
    except GitError:
        pass

    # Delete all agent/* branches
    try:
        result = run_git("branch", "--list", "agent/*", cwd=project_root)
        for branch in result.strip().split("\n"):
            branch = branch.strip().lstrip("* ")
            if branch:
                try:
                    run_git("branch", "-D", branch, cwd=project_root)
                    console.print(f"[dim]Deleted branch {branch}[/dim]")
                except GitError as e:
                    console.print(f"[yellow]Warning: Could not delete {branch}: {e}[/yellow]")
    except GitError:
        pass

    # Clear state
    state_file = project_root / ".crew" / "state.json"
    if state_file.exists():
        state_file.unlink()
        console.print("[dim]Cleared state.json[/dim]")

    # Clear agents from in-memory state
    state.agents.clear()

    # Make sure we're on main
    try:
        run_git("checkout", "main", cwd=project_root)
    except GitError:
        try:
            run_git("checkout", "master", cwd=project_root)
        except GitError:
            pass

    # Git reset --hard
    try:
        run_git("reset", "--hard", commit, cwd=project_root)
        console.print(f"[dim]Reset to {commit}[/dim]")
    except GitError as e:
        print_error(f"Git reset failed: {e}")
        return

    # Git clean -fd (but preserve .tickets)
    try:
        run_git("clean", "-fd", "--exclude=.tickets", cwd=project_root)
        console.print("[dim]Cleaned untracked files (preserved .tickets)[/dim]")
    except GitError as e:
        console.print(f"[yellow]Warning: git clean failed: {e}[/yellow]")

    console.print()
    console.print(f"[green]✓[/green] Reset complete. Fresh start at {commit}")


def cmd_clean(state, args: list[str], project_root: Path) -> None:
    """Wipe all crew state - shutdown agents, remove worktrees, delete state.

    Usage:
        clean          - Clean up everything (prompt for logs deletion)
        clean --logs   - Also delete .crew/logs/
    """
    global _runner

    delete_logs = "--logs" in args

    # If not deleting logs via flag, ask user
    if not delete_logs and (project_root / ".crew" / "logs").exists():
        try:
            response = console.input("[bold]Delete logs too? (y/N): [/bold]")
            delete_logs = response.strip().lower() in ("y", "yes")
        except (KeyboardInterrupt, EOFError):
            print_info("Cancelled")
            return

    console.print()

    # Stop runner if running
    if _runner and _runner.is_running:
        _runner.stop()
        console.print("[dim]Stopped runner[/dim]")

    # Shutdown all agents gracefully
    from crew.git import remove_worktree, run_git, GitError

    for agent in list(state.agents.values()):
        try:
            status = shutdown_agent(agent, state, project_root)
            console.print(f"[dim]Shutdown agent {agent.name} (status: {status})[/dim]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not shutdown {agent.name}: {e}[/yellow]")

    # Remove all worktrees in agents/
    agents_dir = project_root / "agents"
    if agents_dir.exists():
        for worktree_dir in agents_dir.iterdir():
            if worktree_dir.is_dir():
                try:
                    remove_worktree(worktree_dir)
                    console.print(f"[dim]Removed worktree {worktree_dir.name}[/dim]")
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not remove {worktree_dir}: {e}[/yellow]")

    # Prune worktrees
    try:
        run_git("worktree", "prune", cwd=project_root)
    except GitError:
        pass

    # Delete state.json
    state_file = project_root / ".crew" / "state.json"
    if state_file.exists():
        state_file.unlink()
        console.print("[dim]Deleted state.json[/dim]")

    # Clear in-memory state
    state.agents.clear()

    # Optionally delete logs
    if delete_logs:
        logs_dir = project_root / ".crew" / "logs"
        if logs_dir.exists():
            import shutil
            shutil.rmtree(logs_dir)
            console.print("[dim]Deleted logs/[/dim]")

    console.print()
    console.print("[green]✓[/green] Clean complete. All crew state wiped.")


def print_runner_events() -> None:
    """Print any pending events from background runner."""
    global _runner
    if not _runner:
        return

    for event in _runner.drain_events():
        if event.type == "assigned":
            console.print(f"  [cyan]▶[/cyan] {event.agent_name} assigned {event.message}")
        elif event.type == "step":
            console.print(f"  [dim]→ {event.agent_name}:[/dim] {event.message[:50]}...")
        elif event.type == "done":
            console.print(f"  [green]✓[/green] [bold]{event.agent_name}[/bold] completed {event.message}")
        elif event.type == "merged":
            console.print(f"  [green]✓[/green] Merged {event.message}")
        elif event.type == "error":
            console.print(f"  [red]✗[/red] {event.agent_name}: {event.message}")
        elif event.type == "stopped":
            console.print(f"  [blue]ℹ[/blue] {event.message}")


def cmd_peek(state, args: list[str], project_root: Path) -> None:
    """Show agent's recent output."""
    if not args:
        print_error("Usage: peek <name>")
        return

    name = args[0]
    agent = state.get_agent(name)

    if not agent:
        print_error(f"Agent '{name}' not found")
        return

    content = read_log_tail(name, lines=30, project_root=project_root)
    if content:
        print_peek(name, content)
    else:
        print_info(f"No logs for agent '{name}'")


def cmd_ps(state, args: list[str]) -> None:
    """Show all running claude processes."""
    import subprocess as sp
    from rich.table import Table

    try:
        # Get all claude processes with full command line
        result = sp.run(
            ["ps", "-eo", "pid,etime,command"],
            capture_output=True,
            text=True,
        )

        lines = []
        for line in result.stdout.strip().split("\n")[1:]:  # Skip header
            if "claude" in line.lower() and "--print" in line:
                lines.append(line.strip())

        if not lines:
            print_info("No claude processes running")
            return

        table = Table(title="Running Claude Processes", show_header=True, header_style="bold")
        table.add_column("PID", style="cyan", width=8)
        table.add_column("Time", width=10)
        table.add_column("Command", style="dim")

        for line in lines:
            parts = line.split(None, 2)
            if len(parts) >= 3:
                pid, etime, cmd = parts[0], parts[1], parts[2]
                # Truncate long commands
                if len(cmd) > 80:
                    cmd = cmd[:77] + "..."
                table.add_row(pid, etime, cmd)

        console.print(table)
        console.print(f"\n[dim]{len(lines)} process(es)[/dim]")

    except Exception as e:
        print_error(f"Failed to get processes: {e}")


def cmd_logs(state, args: list[str], project_root: Path) -> None:
    """Show logs for an agent or the most recent log.

    Usage:
        logs            - Show tail of most recent step log across all agents
        logs -n 50      - Show last 50 lines of most recent log (default: 30)
        logs <name>     - Show tail of latest log for specific agent
        logs <name> -l  - List all log files for agent
        logs <name> -a  - Show all logs for agent (concatenated)
    """
    # Parse arguments
    lines = 30
    list_files = False
    show_all = False
    agent_name = None

    i = 0
    while i < len(args):
        if args[i] == "-n" and i + 1 < len(args):
            try:
                lines = int(args[i + 1])
                i += 2
                continue
            except ValueError:
                print_error(f"Invalid line count: {args[i + 1]}")
                return
        elif args[i] == "-l":
            list_files = True
            i += 1
        elif args[i] == "-a":
            show_all = True
            i += 1
        elif not agent_name and not args[i].startswith("-"):
            agent_name = args[i]
            i += 1
        else:
            i += 1

    logs_dir = project_root / ".crew" / "logs"

    # If no agent specified, find the most recent log across all agents
    if not agent_name:
        if not logs_dir.exists():
            print_info("No logs found")
            return

        # Find most recent log file across all agent directories
        most_recent_log = None
        most_recent_mtime = 0.0
        most_recent_agent = None

        for agent_dir in logs_dir.iterdir():
            if agent_dir.is_dir():
                for log_file in agent_dir.glob("*.log"):
                    mtime = log_file.stat().st_mtime
                    if mtime > most_recent_mtime:
                        most_recent_mtime = mtime
                        most_recent_log = log_file
                        most_recent_agent = agent_dir.name

        if not most_recent_log:
            print_info("No logs found")
            return

        # Show tail of the most recent log
        content = most_recent_log.read_text()

        # Extract the output section (between --- and === END ===)
        parts = content.split("---\n", 1)
        if len(parts) >= 2:
            output = parts[1].rsplit("=== END ===", 1)[0]
        else:
            output = content

        log_lines = output.strip().split("\n")
        tail = "\n".join(log_lines[-lines:])

        console.print(f"[bold]{most_recent_agent}[/bold] → {most_recent_log.name}")
        console.print(tail)
        return

    # Agent name specified
    log_dir = logs_dir / agent_name

    if not log_dir.exists():
        print_info(f"No logs for agent '{agent_name}'")
        return

    if list_files:
        # List all log files
        console.print(f"[bold]Logs:[/bold] {log_dir}")
        for log_file in sorted(log_dir.glob("*.log")):
            console.print(f"  {log_file.name}")
    elif show_all:
        # Show all logs concatenated
        all_logs = read_all_logs(agent_name, project_root)
        if all_logs:
            console.print(f"[bold]All logs for {agent_name}:[/bold]")
            console.print(all_logs)
        else:
            print_info(f"No logs for agent '{agent_name}'")
    else:
        # Show tail of latest log (default)
        content = read_log_tail(agent_name, lines=lines, project_root=project_root)
        if content:
            console.print(f"[bold]{agent_name}[/bold] (last {lines} lines)")
            console.print(content)
        else:
            print_info(f"No logs for agent '{agent_name}'")


SUMMARIZE_PROMPT = """You are summarizing the work done by a Claude Code agent.

Below are the logs from the agent's session. Provide a concise summary (3-5 bullet points) of:
- What task the agent was working on
- What files were modified or created
- Key changes or features implemented
- Any issues encountered or left unresolved

Be specific and factual. Use file names and function names where relevant.

=== LOGS ===
{logs}
=== END LOGS ===

Provide your summary now:"""


HAIKU_SUMMARY_PROMPT = """Summarize this agent's recent work in 2-3 short bullet points.
Focus on: what was done, which files changed, current status.
Be very brief - each bullet should be under 60 chars.

=== LOGS ===
{logs}
=== END LOGS ===

Summary:"""


def generate_haiku_summary(agent_name: str, project_root: Path) -> tuple[str | None, float]:
    """Generate a brief summary using Haiku model.

    Args:
        agent_name: Name of the agent to summarize
        project_root: Project root path

    Returns:
        Tuple of (summary text or None, cost in USD)
    """
    from crew.runner import run_claude, generate_session_id
    from datetime import datetime

    logs = read_all_logs(agent_name, project_root)

    if not logs:
        return None, 0.0

    # Keep only recent logs (last 10k chars for Haiku efficiency)
    max_log_size = 10000
    if len(logs) > max_log_size:
        logs = "... (earlier logs truncated) ...\n\n" + logs[-max_log_size:]

    try:
        response = run_claude(
            HAIKU_SUMMARY_PROMPT.format(logs=logs),
            cwd=project_root,
            session=generate_session_id(),
            is_new_session=True,
            timeout=30,
            model="haiku",
        )

        return response["result"].strip(), response["cost_usd"]

    except Exception:
        return None, 0.0


def cmd_summarize(state, args: list[str], project_root: Path) -> None:
    """Summarize an agent's work using Claude.

    Usage: summarize <name>

    Reads all logs for the agent and uses Claude to generate a
    concise summary of what was accomplished.
    """
    from crew.runner import run_claude, generate_session_id

    if not args:
        print_error("Usage: summarize <name>")
        return

    name = args[0]
    agent = state.get_agent(name)

    if not agent:
        print_error(f"Agent '{name}' not found")
        return

    # Read all logs for the agent
    logs = read_all_logs(name, project_root)

    if not logs:
        print_info(f"No logs for agent '{name}'")
        return

    # Truncate logs if too long (keep last ~50k chars to stay within context)
    max_log_size = 50000
    if len(logs) > max_log_size:
        logs = "... (earlier logs truncated) ...\n\n" + logs[-max_log_size:]

    console.print(f"[dim]Generating summary for {name}...[/dim]")

    try:
        # Use Claude to generate summary
        response = run_claude(
            SUMMARIZE_PROMPT.format(logs=logs),
            cwd=project_root,
            session=generate_session_id(),
            is_new_session=True,
            timeout=60,
        )

        summary = response["result"]

        # Display the summary
        from rich.panel import Panel
        console.print()
        console.print(Panel(
            summary,
            title=f"[bold]Summary: {name}[/bold]" + (f" ({agent.task})" if agent.task else ""),
            border_style="cyan",
            padding=(1, 2),
        ))

        # Show cost info
        cost = response["cost_usd"]
        tokens = response["input_tokens"] + response["output_tokens"]
        console.print(f"[dim]Summary cost: {tokens:,} tokens, ${cost:.4f}[/dim]")

    except Exception as e:
        print_error(f"Failed to generate summary: {e}")


def cmd_refresh_summary(state, args: list[str], project_root: Path) -> None:
    """Refresh Haiku summary for an agent or all agents.

    Usage:
        refresh-summary <name>    Refresh summary for one agent
        refresh-summary --all     Refresh summaries for all agents

    Generates a brief summary using Haiku (fast and cheap) and
    caches it for display in the dashboard.
    """
    from datetime import datetime

    if not args:
        print_error("Usage: refresh-summary <name> or refresh-summary --all")
        return

    if args[0] == "--all":
        agents_to_refresh = [a for a in state.agents.values() if a.status in ("ready", "working", "done")]
    else:
        name = args[0]
        agent = state.get_agent(name)
        if not agent:
            print_error(f"Agent '{name}' not found")
            return
        agents_to_refresh = [agent]

    if not agents_to_refresh:
        print_info("No agents to refresh")
        return

    total_cost = 0.0
    for agent in agents_to_refresh:
        console.print(f"[dim]Refreshing summary for {agent.name}...[/dim]")
        summary, cost = generate_haiku_summary(agent.name, project_root)
        if summary:
            agent.summary = summary
            agent.summary_updated_at = datetime.now()
            total_cost += cost
            console.print(f"[green]✓[/green] {agent.name}: summary updated")
        else:
            console.print(f"[yellow]○[/yellow] {agent.name}: no logs or failed")

    state.save()
    if total_cost > 0:
        console.print(f"[dim]Total summary cost: ${total_cost:.4f}[/dim]")


def cmd_kill(state, args: list[str], project_root: Path) -> None:
    """Stop an agent, release its task, and remove worktree.

    Committed work stays on the agent's branch. The task becomes
    available for another agent to pick up.
    """
    if not args:
        print_error("Usage: kill <name>")
        return

    name = args[0]
    agent = state.get_agent(name)

    if not agent:
        print_error(f"Agent '{name}' not found")
        return

    old_task = agent.task
    old_worktree = agent.worktree

    # Remove worktree if it exists (committed work stays on branch)
    if old_worktree and old_worktree.exists():
        try:
            remove_worktree(old_worktree)
        except Exception as e:
            print_error(f"Failed to remove worktree: {e}")
            # Continue anyway - reset agent state

    # Reset agent to clean idle state
    agent.status = "idle"
    agent.task = None
    agent.worktree = None
    agent.branch = ""
    agent.session = ""
    save_state(state, project_root)

    if old_task:
        print_info(f"Killed agent '{name}', released task {old_task}. Ready for new work.")
    else:
        print_info(f"Killed agent '{name}'. Ready for new work.")


def cmd_cleanup(state, args: list[str], project_root: Path) -> None:
    """Remove agent and worktree."""
    if not args:
        print_error("Usage: cleanup <name>")
        return

    name = args[0]
    agent = state.get_agent(name)

    if not agent:
        print_error(f"Agent '{name}' not found")
        return

    try:
        cleanup_agent(agent, state, merge=False, project_root=project_root)
        print_info(f"Removed agent '{name}' and worktree")
    except Exception as e:
        print_error(f"Cleanup failed: {e}")


def cmd_merge(state, args: list[str], project_root: Path) -> None:
    """Merge agent's branch to main."""
    if not args:
        print_error("Usage: merge <name>")
        return

    name = args[0]
    agent = state.get_agent(name)

    if not agent:
        print_error(f"Agent '{name}' not found")
        return

    try:
        cleanup_agent(agent, state, merge=True, project_root=project_root)
        print_agent_merged(agent)
    except Exception as e:
        print_error(f"Merge failed: {e}")


def cmd_ready(state, args: list[str]) -> None:
    """Show ready tickets with agent assignment indicators."""
    output = run_tk("ready")
    if not output.strip():
        console.print("[dim]No ready work.[/dim]")
        return

    # Build map of task_id -> agent_name for assigned tasks
    assigned = {}
    for agent in state.agents.values():
        if agent.task:
            assigned[agent.task] = agent.name

    # Process each line and add assignment markers
    in_progress = []
    available = []

    for line in output.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split()
        if not parts:
            continue
        task_id = parts[0]
        if task_id in assigned:
            agent_name = assigned[task_id]
            in_progress.append((task_id, line, agent_name))
        else:
            available.append(line)

    # Display in progress tickets first
    if in_progress:
        console.print("[bold yellow]In Progress[/bold yellow]")
        for task_id, line, agent_name in in_progress:
            # Replace the task_id with a marked version
            # Use \[ to escape opening bracket in Rich markup
            marked_line = line.replace(task_id, f"{task_id} \\[{agent_name}]", 1)
            console.print(f"  {marked_line}")
        console.print()

    # Display available tickets
    if available:
        console.print("[bold green]Available[/bold green]")
        for line in available:
            console.print(f"  {line}")
    elif not in_progress:
        console.print("[dim]No ready work.[/dim]")


def cmd_new(state, args: list[str]) -> None:
    """Create a new ticket, optionally with dependencies.

    Usage:
        new <title>
        new <title> --dep <id> [--dep <id2>...]
        new <title> --after <id>  (alias for --dep)
    """
    if not args:
        print_error("Usage: new <title> [--dep <id>...]")
        return

    # Parse args for --dep/--after flags
    title_parts = []
    deps = []
    i = 0
    while i < len(args):
        if args[i] in ("--dep", "--after", "-d"):
            if i + 1 < len(args):
                deps.append(args[i + 1])
                i += 2
            else:
                print_error(f"{args[i]} requires a ticket ID")
                return
        else:
            title_parts.append(args[i])
            i += 1

    if not title_parts:
        print_error("Usage: new <title> [--dep <id>...]")
        return

    title = " ".join(title_parts)

    # Create the ticket
    output = run_tk("create", title)
    console.print(output)

    # Extract ticket ID from output (tk create prints the ID)
    ticket_id = output.strip().split()[0] if output.strip() else None

    # Add dependencies if any
    if ticket_id and deps:
        for dep_id in deps:
            dep_output = run_tk("dep", ticket_id, dep_id)
            if dep_output.strip():
                console.print(f"  [dim]{dep_output.strip()}[/dim]")


def cmd_dep(state, args: list[str]) -> None:
    """Add a dependency between tickets.

    Usage: dep <ticket> <depends-on>
    Means: <ticket> depends on <depends-on> (ticket is blocked until depends-on closes)
    """
    if len(args) < 2:
        print_error("Usage: dep <ticket> <depends-on>")
        print_info("Example: dep mp-abc mp-xyz  (abc is blocked by xyz)")
        return

    ticket_id = args[0]
    dep_id = args[1]

    output = run_tk("dep", ticket_id, dep_id)
    console.print(output if output.strip() else f"[green]✓[/green] {ticket_id} now depends on {dep_id}")


def cmd_assign(state, args: list[str], project_root: Path) -> None:
    """Assign a ticket to an agent."""
    if len(args) < 2:
        print_error("Usage: assign <name> <ticket-id>")
        return

    name, task_id = args[0], args[1]
    agent = state.get_agent(name)

    if not agent:
        print_error(f"Agent '{name}' not found")
        return

    agent.task = task_id
    save_state(state, project_root)
    print_info(f"Assigned {task_id} to {name}")


def cmd_queue(state, args: list[str]) -> None:
    """Show full tk dependency pipeline with scheduling waves.

    Displays tickets organized into waves based on dependencies:
    - Ready now: No unresolved dependencies, can start immediately
    - Next: Grouped by what blocks them (will be ready when blocker completes)
    - Later: Multiple levels of dependencies away
    - Blocked: Has dependencies that are themselves blocked or stuck
    """
    import json
    from collections import defaultdict

    # Get all tickets as JSON from tk query
    try:
        result = subprocess.run(
            ["tk", "query", "--json"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print_error(f"tk query failed: {result.stderr}")
            return
        tickets_data = json.loads(result.stdout) if result.stdout.strip() else []
    except FileNotFoundError:
        print_error("tk command not found. Is ticket installed?")
        return
    except json.JSONDecodeError as e:
        print_error(f"Failed to parse tk query output: {e}")
        return

    if not tickets_data:
        console.print("[dim]No tickets in queue.[/dim]")
        return

    # Build ticket lookup and dependency graph
    tickets = {}  # id -> ticket data
    for t in tickets_data:
        tid = t.get("id", "")
        if tid:
            tickets[tid] = t

    # Only consider open tickets
    open_tickets = {tid: t for tid, t in tickets.items() if t.get("status") == "open"}

    if not open_tickets:
        console.print("[dim]No open tickets in queue.[/dim]")
        return

    # Build reverse dependency map: blocker_id -> list of tickets blocked by it
    blocked_by = defaultdict(list)  # blocker_id -> [tickets waiting on it]
    for tid, t in open_tickets.items():
        deps = t.get("deps", [])
        for dep_id in deps:
            if dep_id in open_tickets:  # Only count open deps as blocking
                blocked_by[dep_id].append(tid)

    # Compute waves using topological sort (Kahn's algorithm)
    # Wave 0 = ready now (no open deps)
    # Wave N = tickets whose deps are all in waves < N
    waves = {}  # ticket_id -> wave_number
    remaining = set(open_tickets.keys())

    wave_num = 0
    while remaining:
        # Find tickets with no unresolved open dependencies
        ready_this_wave = []
        for tid in remaining:
            deps = open_tickets[tid].get("deps", [])
            # A ticket is ready if all its deps are either closed or in previous waves
            open_deps = [d for d in deps if d in open_tickets and d not in waves]
            if not open_deps:
                ready_this_wave.append(tid)

        if not ready_this_wave:
            # Remaining tickets are in a cycle - mark as blocked
            for tid in remaining:
                waves[tid] = -1  # -1 indicates blocked/cyclic
            break

        for tid in ready_this_wave:
            waves[tid] = wave_num
            remaining.discard(tid)

        wave_num += 1

    # Group tickets by wave
    wave_tickets = defaultdict(list)
    for tid, wave in waves.items():
        wave_tickets[wave].append(tid)

    # Display results
    from rich.table import Table

    # Ready now (wave 0)
    if 0 in wave_tickets:
        console.print("[bold green]Ready now[/bold green]")
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("ID", style="cyan")
        table.add_column("Title")
        for tid in sorted(wave_tickets[0]):
            t = open_tickets[tid]
            # Extract title from ticket (first line of body after frontmatter)
            title = t.get("title", tid)
            table.add_row(tid, title)
        console.print(table)
        console.print()

    # Next (wave 1) - grouped by what blocks them
    if 1 in wave_tickets:
        console.print("[bold yellow]Next[/bold yellow] (waiting on ready tickets)")
        # Group by blockers
        by_blocker = defaultdict(list)
        for tid in wave_tickets[1]:
            deps = open_tickets[tid].get("deps", [])
            open_deps = [d for d in deps if d in open_tickets]
            blocker_key = ", ".join(sorted(open_deps)) if open_deps else "none"
            by_blocker[blocker_key].append(tid)

        for blocker, tids in sorted(by_blocker.items()):
            console.print(f"  [dim]blocked by {blocker}:[/dim]")
            for tid in sorted(tids):
                t = open_tickets[tid]
                title = t.get("title", tid)
                console.print(f"    [cyan]{tid}[/cyan] {title}")
        console.print()

    # Later (waves 2+)
    later_waves = [w for w in wave_tickets.keys() if w >= 2]
    if later_waves:
        console.print("[bold blue]Later[/bold blue] (multiple dependencies away)")
        for wave in sorted(later_waves):
            console.print(f"  [dim]wave {wave}:[/dim]")
            for tid in sorted(wave_tickets[wave]):
                t = open_tickets[tid]
                title = t.get("title", tid)
                deps = t.get("deps", [])
                open_deps = [d for d in deps if d in open_tickets]
                dep_str = f" [dim](deps: {', '.join(open_deps)})[/dim]" if open_deps else ""
                console.print(f"    [cyan]{tid}[/cyan] {title}{dep_str}")
        console.print()

    # Blocked (wave -1, cyclic dependencies)
    if -1 in wave_tickets:
        console.print("[bold red]Blocked[/bold red] (cyclic or stuck dependencies)")
        for tid in sorted(wave_tickets[-1]):
            t = open_tickets[tid]
            title = t.get("title", tid)
            deps = t.get("deps", [])
            open_deps = [d for d in deps if d in open_tickets]
            dep_str = f" [dim](deps: {', '.join(open_deps)})[/dim]" if open_deps else ""
            console.print(f"  [cyan]{tid}[/cyan] {title}{dep_str}")
        console.print()

    # Summary
    total = len(open_tickets)
    ready = len(wave_tickets.get(0, []))
    blocked = len(wave_tickets.get(-1, []))
    console.print(f"[dim]{total} open tickets: {ready} ready, {total - ready - blocked} pending, {blocked} blocked[/dim]")


def render_dashboard(state, project_root: Path, runner_active: bool = False, show_summaries: bool = False) -> "Group":
    """Render dashboard content as a Rich renderable.

    Args:
        state: Current crew state
        project_root: Project root path
        runner_active: Whether the background runner is active
        show_summaries: Whether to show AI-generated summaries for working agents

    Returns:
        A Rich Group containing all dashboard elements
    """
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.console import Group

    renderables = []

    # Runner status
    if runner_active:
        working = len([a for a in state.agents.values() if a.status in ("ready", "working")])
        idle = len([a for a in state.agents.values() if a.status == "idle"])
        if working:
            renderables.append(Text.from_markup(f"[bold green]● Runner active[/bold green] ({working} working, {idle} idle)"))
        else:
            renderables.append(Text.from_markup(f"[bold green]● Runner active[/bold green] ({idle} idle)"))
    else:
        renderables.append(Text.from_markup("[dim]○ Runner stopped[/dim]"))

    # Session cost total
    total_tokens = sum(a.total_input_tokens + a.total_output_tokens for a in state.agents.values())
    total_cost = sum(a.total_cost_usd for a in state.agents.values())
    renderables.append(Text.from_markup(f"[bold]Session:[/bold] {total_tokens:,} tokens, ${total_cost:.4f}"))
    renderables.append(Text(""))  # Empty line

    # Agent table with columns: NAME, TASK, STATUS, STEPS, TOKENS, COST
    if state.agents:
        table = Table(show_header=True, header_style="bold")
        table.add_column("NAME")
        table.add_column("TASK")
        table.add_column("STATUS")
        table.add_column("STEPS", justify="right")
        table.add_column("TOKENS", justify="right")
        table.add_column("COST", justify="right")

        for agent in state.agents.values():
            # Status with color
            status_styles = {
                "idle": "dim",
                "ready": "yellow",
                "working": "green",
                "done": "blue",
                "stuck": "red",
            }
            style = status_styles.get(agent.status, "")

            # Calculate tokens
            tokens = agent.total_input_tokens + agent.total_output_tokens

            table.add_row(
                agent.name,
                agent.task or "-",
                Text(agent.status, style=style),
                str(agent.step_count),
                f"{tokens:,}" if tokens else "-",
                f"${agent.total_cost_usd:.4f}" if agent.total_cost_usd else "-",
            )

        renderables.append(table)

        # Log tail panels for working agents
        working_agents = [a for a in state.agents.values() if a.status in ("ready", "working")]
        if working_agents:
            renderables.append(Text(""))  # Empty line
            colors = ["cyan", "green", "yellow", "magenta", "blue", "red"]
            for i, agent in enumerate(working_agents):
                color = colors[i % len(colors)]
                content = read_log_tail(agent.name, lines=5, project_root=project_root)
                if content:
                    # Truncate long lines for display
                    lines = content.split("\n")
                    truncated = "\n".join(
                        line[:100] + "..." if len(line) > 100 else line
                        for line in lines[-5:]
                    )
                    header = f"● {agent.name}"
                    if agent.task:
                        header += f" → {agent.task}"
                    renderables.append(Panel(
                        truncated,
                        title=f"[bold {color}]{header}[/bold {color}]",
                        border_style=color,
                        padding=(0, 1),
                    ))

        # Git status panels for working agents
        from crew.git import run_git, GitError

        working_agents = [a for a in state.agents.values() if a.worktree and a.status in ("ready", "working")]
        if working_agents:
            renderables.append(Text(""))  # Empty line
            for i, agent in enumerate(working_agents):
                color = colors[i % len(colors)]

                # Get git status --short
                try:
                    status = run_git("status", "--short", cwd=agent.worktree)
                except (GitError, OSError):
                    status = "[dim]Unable to read worktree[/dim]"

                # Get git diff --stat
                try:
                    diff_stat = run_git("diff", "--stat", "HEAD", cwd=agent.worktree)
                except (GitError, OSError):
                    diff_stat = ""

                # Build content
                content_parts = []
                if status.strip():
                    content_parts.append(f"[bold]Status:[/bold]\n{status}")
                if diff_stat.strip():
                    content_parts.append(f"[bold]Diff:[/bold]\n{diff_stat}")

                content = "\n\n".join(content_parts) if content_parts else "[dim]No changes yet[/dim]"

                # Build header
                header = f"● {agent.name}"
                if agent.task:
                    header += f" → {agent.task}"

                renderables.append(Panel(
                    content,
                    title=f"[bold {color}]{header}[/bold {color}]",
                    border_style=color,
                    padding=(0, 1),
                ))

        # Summary panels for agents with cached summaries
        agents_with_summaries = [a for a in state.agents.values() if a.summary]
        if agents_with_summaries:
            renderables.append(Text(""))  # Empty line
            renderables.append(Text.from_markup("[bold]Summaries:[/bold]"))
            for i, agent in enumerate(agents_with_summaries):
                color = colors[i % len(colors)]

                # Build header with age indicator
                header = f"✦ {agent.name}"
                if agent.task:
                    header += f" → {agent.task}"

                renderables.append(Panel(
                    agent.summary,
                    title=f"[bold {color}]{header}[/bold {color}]",
                    border_style=color,
                    padding=(0, 1),
                ))

        # AI summary panels for working agents (when -s flag is used)
        if show_summaries:
            from crew.runner import run_claude, generate_session_id
            working_agents = [a for a in state.agents.values() if a.status in ("ready", "working")]
            if working_agents:
                renderables.append(Text(""))  # Empty line
                renderables.append(Text.from_markup("[bold]AI Summaries[/bold]"))
                for i, agent in enumerate(working_agents):
                    color = colors[i % len(colors)]
                    logs = read_all_logs(agent.name, project_root)
                    if logs:
                        # Truncate logs if too long
                        max_log_size = 50000
                        if len(logs) > max_log_size:
                            logs = "... (earlier logs truncated) ...\n\n" + logs[-max_log_size:]
                        try:
                            response = run_claude(
                                SUMMARIZE_PROMPT.format(logs=logs),
                                cwd=project_root,
                                session=generate_session_id(),
                                is_new_session=True,
                                timeout=60,
                            )
                            summary = response["result"]
                            cost = response["cost_usd"]
                            tokens = response["input_tokens"] + response["output_tokens"]
                            footer = f"[dim]{tokens:,} tokens, ${cost:.4f}[/dim]"
                        except Exception as e:
                            summary = f"[dim]Failed to generate summary: {e}[/dim]"
                            footer = ""
                    else:
                        summary = "[dim]No logs available[/dim]"
                        footer = ""

                    header = f"● {agent.name}"
                    if agent.task:
                        header += f" → {agent.task}"
                    renderables.append(Panel(
                        summary + ("\n\n" + footer if footer else ""),
                        title=f"[bold {color}]{header}[/bold {color}]",
                        border_style=color,
                        padding=(0, 1),
                    ))
    else:
        renderables.append(Text.from_markup("[dim]No agents.[/dim]"))

    return Group(*renderables)


def cmd_dashboard(state, args: list[str], project_root: Path) -> None:
    """Show dashboard with runner status, costs, and agent table.

    Usage:
        dashboard         Show dashboard once
        dashboard -l      Live mode: auto-refresh every 2 seconds, press 'q' to exit
        dashboard --live  Same as -l
        dashboard -s      Show AI-generated summaries for working agents
        dashboard --summary  Same as -s
    """
    global _runner

    # Check for live mode flag
    live_mode = "-l" in args or "--live" in args
    # Check for summary flag
    show_summaries = "-s" in args or "--summary" in args

    if live_mode:
        _run_live_dashboard(state, project_root, show_summaries=show_summaries)
    else:
        # Print any pending events first
        print_runner_events()

        # Render and print dashboard
        runner_active = _runner is not None and _runner.is_running
        dashboard = render_dashboard(state, project_root, runner_active, show_summaries=show_summaries)
        console.print(dashboard)


def _run_live_dashboard(state, project_root: Path, show_summaries: bool = False) -> None:
    """Run the dashboard in live mode with auto-refresh.

    Refreshes every 2 seconds. Press 'q' to exit back to REPL.
    """
    global _runner
    import sys
    import select
    import termios
    import tty
    from rich.live import Live

    # Save terminal settings
    old_settings = termios.tcgetattr(sys.stdin)

    try:
        # Set terminal to raw mode for single key detection
        tty.setcbreak(sys.stdin.fileno())

        console.print("[dim]Live dashboard mode. Press 'q' to exit.[/dim]")
        console.print()

        with Live(console=console, refresh_per_second=0.5, screen=False) as live:
            while True:
                # Check for 'q' key press (non-blocking)
                if select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.read(1)
                    if key.lower() == 'q':
                        break

                # Drain and display any pending runner events
                if _runner:
                    for event in _runner.drain_events():
                        if event.type == "assigned":
                            console.print(f"  [cyan]▶[/cyan] {event.agent_name} assigned {event.message}")
                        elif event.type == "step":
                            console.print(f"  [dim]→ {event.agent_name}:[/dim] {event.message[:50]}...")
                        elif event.type == "done":
                            console.print(f"  [green]✓[/green] [bold]{event.agent_name}[/bold] completed {event.message}")
                        elif event.type == "merged":
                            console.print(f"  [green]✓[/green] Merged {event.message}")
                        elif event.type == "error":
                            console.print(f"  [red]✗[/red] {event.agent_name}: {event.message}")
                        elif event.type == "stopped":
                            console.print(f"  [blue]ℹ[/blue] {event.message}")

                # Update the live display
                runner_active = _runner is not None and _runner.is_running
                dashboard = render_dashboard(state, project_root, runner_active, show_summaries=show_summaries)
                live.update(dashboard)

                # Sleep for 2 seconds (the refresh interval)
                import time
                time.sleep(2)

    except KeyboardInterrupt:
        pass
    finally:
        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        console.print()
        console.print("[dim]Exited live mode.[/dim]")


def handle_command(line: str, state, project_root: Path) -> bool:
    """Handle a command line. Returns True to continue, False to quit."""
    global _read_only_mode

    parts = line.strip().split()
    if not parts:
        return True

    cmd = parts[0].lower()
    args = parts[1:]

    # Block modifying commands in read-only mode
    if _read_only_mode and cmd in _MODIFYING_COMMANDS:
        print_error(f"Cannot run '{cmd}' in read-only mode. Another crew instance is running.")
        return True

    if cmd in ("quit", "exit"):
        return False
    elif cmd in ("queue", "q"):
        cmd_queue(state, args)
    elif cmd in ("h", "help"):
        print_help()
    elif cmd in ("dashboard", "d", "s", "status", "dash", "st"):
        cmd_dashboard(state, args, project_root)
    elif cmd == "spawn":
        cmd_spawn(state, args, project_root)
    elif cmd in ("run", "work"):
        cmd_run(state, args, project_root)
    elif cmd == "stop":
        cmd_stop(state, args, project_root)
    elif cmd == "ps":
        cmd_ps(state, args)
    elif cmd == "peek":
        cmd_peek(state, args, project_root)
    elif cmd == "logs":
        cmd_logs(state, args, project_root)
    elif cmd in ("summarize", "sum"):
        cmd_summarize(state, args, project_root)
    elif cmd == "refresh-summary":
        cmd_refresh_summary(state, args, project_root)
    elif cmd == "kill":
        cmd_kill(state, args, project_root)
    elif cmd == "cleanup":
        cmd_cleanup(state, args, project_root)
    elif cmd == "merge":
        cmd_merge(state, args, project_root)
    elif cmd in ("r", "ready"):
        cmd_ready(state, args)
    elif cmd == "new":
        cmd_new(state, args)
    elif cmd == "dep":
        cmd_dep(state, args)
    elif cmd == "assign":
        cmd_assign(state, args, project_root)
    elif cmd == "reset":
        cmd_reset(state, args, project_root)
    elif cmd == "clean":
        cmd_clean(state, args, project_root)
    elif cmd == "tui":
        from crew.tui import main as tui_main
        tui_main()
    else:
        print_error(f"Unknown command: {cmd}. Type 'help' for commands.")

    return True


def recover_session(state, project_root: Path) -> bool:
    """Recover session state on startup.

    Detects prior session state from .crew/state.json and reconciles it:
    - Displays resume message showing agents and their status
    - Validates worktrees exist on disk
    - Warns about orphaned worktrees in agents/
    - Resets agents to idle if their worktree is missing
    - Reconciles working agents that were mid-step when process died

    Args:
        state: The loaded state
        project_root: Project root path

    Returns:
        True if there was a prior session to recover, False otherwise
    """
    if not state.agents:
        return False

    console.print()
    console.print("[bold]Recovering prior session...[/bold]")
    console.print()

    # Get actual worktrees from git
    try:
        git_worktrees = get_worktree_list()
        # Extract paths from worktree list, filtering out bare repos
        actual_worktree_paths = {
            Path(wt["path"]) for wt in git_worktrees
            if not wt.get("bare") and "path" in wt
        }
    except Exception as e:
        print_warning(f"Could not list worktrees: {e}")
        actual_worktree_paths = set()

    # Track orphaned worktrees (in agents/ but not in state)
    agents_dir = project_root / "agents"
    orphaned_worktrees = []
    if agents_dir.exists():
        state_worktree_paths = {
            Path(a.worktree) for a in state.agents.values()
            if a.worktree
        }
        for subdir in agents_dir.iterdir():
            if subdir.is_dir():
                if subdir not in state_worktree_paths:
                    orphaned_worktrees.append(subdir)

    # Display agents and their status
    console.print("[bold]Agents:[/bold]")
    actions_taken = []

    for agent in list(state.agents.values()):
        status_icon = get_status_icon(agent.status)
        task_info = f" → {agent.task}" if agent.task else ""

        # Check if worktree exists on disk
        worktree_exists = (
            agent.worktree
            and Path(agent.worktree).exists()
            and Path(agent.worktree) in actual_worktree_paths
        )

        # Handle different recovery scenarios
        if agent.status in ("ready", "working"):
            if not worktree_exists and agent.worktree:
                # Worktree missing but agent expects it - reset to idle
                old_status = agent.status
                old_task = agent.task
                agent.status = "idle"
                agent.worktree = None
                agent.branch = ""
                agent.task = None
                agent.session = ""
                agent.step_count = 0
                agent.last_step_at = None
                console.print(f"  {status_icon} [bold]{agent.name}[/bold]{task_info} [dim]({old_status})[/dim]")
                actions_taken.append(f"Reset {agent.name} to idle (worktree missing)")
            else:
                # Worktree exists - check if it has uncommitted changes
                worktree_path = Path(agent.worktree)
                console.print(f"  {status_icon} [bold]{agent.name}[/bold]{task_info} [dim]({agent.status})[/dim]")

                if has_uncommitted_changes(worktree_path):
                    # Dirty worktree - simpler recovery: remove worktree and reset to idle
                    # The ticket stays open and can be re-assigned
                    old_task = agent.task
                    remove_worktree(worktree_path)
                    agent.status = "idle"
                    agent.worktree = None
                    agent.branch = ""
                    agent.task = None
                    agent.session = ""
                    agent.step_count = 0
                    agent.last_step_at = None
                    actions_taken.append(f"Reset {agent.name} to idle (dirty worktree removed, ticket {old_task} stays open)")
                else:
                    # Clean worktree - reconcile based on logs
                    work_status = shutdown_agent(agent, state, project_root)
                    if work_status == "done":
                        # Agent completed their task - run complete_task to merge, close ticket, return to idle
                        task_id = agent.task
                        branch = agent.branch
                        try:
                            complete_task(agent, state, project_root, console=console)
                            actions_taken.append(f"Completed {agent.name} (merged {branch}, closed {task_id})")
                        except Exception as e:
                            actions_taken.append(f"Failed to complete {agent.name}: {e}")
                            # Leave agent as done so user can manually handle
                    elif work_status == "partial":
                        # Partial work but clean worktree - reset to idle, don't try to recover
                        # The committed work stays on the branch but agent is reset
                        agent.status = "idle"
                        agent.worktree = None
                        agent.branch = ""
                        agent.task = None
                        agent.session = ""
                        agent.step_count = 0
                        agent.last_step_at = None
                        remove_worktree(worktree_path)
                        actions_taken.append(f"Reset {agent.name} to idle (partial work, ticket stays open)")
                    elif work_status == "nothing":
                        # No work done - reset to idle (simpler: don't try to keep as "ready")
                        agent.status = "idle"
                        agent.worktree = None
                        agent.branch = ""
                        agent.task = None
                        agent.session = ""
                        agent.step_count = 0
                        agent.last_step_at = None
                        remove_worktree(worktree_path)
                        actions_taken.append(f"Reset {agent.name} to idle (no work done, ticket stays open)")
        elif agent.status == "idle":
            # Idle agents are fine, just display
            console.print(f"  {status_icon} [bold]{agent.name}[/bold] [dim](idle)[/dim]")
        elif agent.status == "done":
            # Done agents - need to call complete_task to finish the lifecycle
            if worktree_exists:
                console.print(f"  {status_icon} [bold]{agent.name}[/bold]{task_info} [dim](done, pending merge)[/dim]")
                # Complete the task (merge, cleanup, return to idle)
                task_id = agent.task
                branch = agent.branch
                try:
                    complete_task(agent, state, project_root, console=console)
                    actions_taken.append(f"Completed {agent.name} (merged {branch}, closed {task_id})")
                except Exception as e:
                    actions_taken.append(f"Failed to complete {agent.name}: {e}")
            else:
                # Worktree gone but agent marked done - preserve done status
                # Work is committed to the branch and can be merged via 'merge <name>' command
                console.print(f"  {status_icon} [bold]{agent.name}[/bold]{task_info} [dim](done, worktree missing)[/dim]")
                console.print(f"    [dim]Work is on branch {agent.branch}. Use 'merge {agent.name}' to complete.[/dim]")
                actions_taken.append(f"Preserved {agent.name} as done (use 'merge {agent.name}' to complete)")
        elif agent.status == "stuck":
            console.print(f"  {status_icon} [bold]{agent.name}[/bold]{task_info} [dim](stuck)[/dim]")
        else:
            console.print(f"  {status_icon} [bold]{agent.name}[/bold]{task_info} [dim]({agent.status})[/dim]")

    console.print()

    # Warn about orphaned worktrees
    if orphaned_worktrees:
        print_warning("Orphaned worktrees found in agents/ (not tracked in state):")
        for wt in orphaned_worktrees:
            console.print(f"  [dim]{wt}[/dim]")
        console.print("  [dim]Use 'cleanup' or remove manually[/dim]")
        console.print()

    # Display actions taken
    if actions_taken:
        console.print("[bold]Recovery actions:[/bold]")
        for action in actions_taken:
            console.print(f"  [cyan]•[/cyan] {action}")
        console.print()

    # Save reconciled state
    save_state(state, project_root)

    # Summary
    working = len([a for a in state.agents.values() if a.status in ("ready", "working")])
    idle = len([a for a in state.agents.values() if a.status == "idle"])
    done = len([a for a in state.agents.values() if a.status == "done"])

    if working:
        console.print(f"[green]Ready to resume[/green]: {working} working, {idle} idle, {done} done")
        print_info("Use 'run' to continue")
    elif idle:
        console.print(f"[blue]Session restored[/blue]: {idle} idle agents ready for work")
        print_info("Use 'run' to auto-assign tasks")
    else:
        console.print(f"[blue]Session restored[/blue]: {len(state.agents)} agent(s)")

    console.print()
    return True


def main() -> None:
    """Main entry point."""
    global _read_only_mode, _state, _project_root, _runner, _shutting_down

    project_root = Path.cwd()
    _project_root = project_root

    # Ensure .crew directory exists
    ensure_crew_dir(project_root)

    # Check for existing instance and acquire PID lock
    if check_and_acquire_lock(project_root):
        _read_only_mode = False
    else:
        _read_only_mode = True
        print_warning("Another crew instance is running. Entering read-only mode.")

    # Load state
    state = load_state(project_root)
    _state = state

    # Register signal handlers for graceful shutdown
    # SIGINT = Ctrl-C, SIGTERM = kill command
    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

    # Set up prompt session with history
    history_file = project_root / ".crew" / "history"
    session = PromptSession(
        history=FileHistory(str(history_file)),
        auto_suggest=AutoSuggestFromHistory(),
    )

    # Print banner
    print_banner()

    # Recover prior session if any
    recover_session(state, project_root)

    # REPL loop
    while True:
        try:
            # Print any pending events from background runner
            print_runner_events()

            line = session.prompt(get_prompt(state))
            if not handle_command(line, state, project_root):
                break
        except KeyboardInterrupt:
            # Signal handler takes care of graceful shutdown
            # This exception may still be raised by prompt_toolkit
            if _shutting_down:
                break
            # If not shutting down, just stop the runner and continue
            if _runner and _runner.is_running:
                _runner.stop()
                print_runner_events()
                print_info("Stopped runner")
            continue
        except EOFError:
            break

    # Mark that we're shutting down to prevent signal handler re-entry
    _shutting_down = True

    # Stop runner on exit
    if _runner and _runner.is_running:
        _runner.stop()

    # Shutdown working agents and save state
    for agent in list(state.agents.values()):
        if agent.status in ("ready", "working"):
            try:
                shutdown_agent(agent, state, project_root)
            except Exception:
                pass  # Best effort on shutdown

    save_state(state, project_root)

    # Clean up PID file on exit (only if we're not in read-only mode)
    if not _read_only_mode:
        remove_pid_file(project_root)

    console.print("[dim]bye[/dim]")


if __name__ == "__main__":
    main()
