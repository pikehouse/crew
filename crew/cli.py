"""CLI and REPL interface for crew."""

from __future__ import annotations

import asyncio
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
)
from crew.runner import spawn_agent, spawn_worker, step_agent, cleanup_agent, assign_task, complete_task, get_ready_tasks
from crew.crew_logging import read_log_tail

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
        # Lock for thread-safe operations
        self._lock = threading.Lock()

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
        """Stop the background runner."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
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

        # Get ready tasks that aren't already assigned
        assigned = self._get_assigned_tasks()
        ready_tasks = [t for t in get_ready_tasks() if t not in assigned]

        # Assign tasks to idle agents
        for agent, task_id in zip(idle_agents, ready_tasks):
            try:
                assign_task(agent, task_id, self.state, self.project_root)
                self.events.put(RunnerEvent("assigned", agent.name, task_id))
            except Exception as e:
                self.events.put(RunnerEvent("error", agent.name, f"Failed to assign {task_id}: {e}"))

    def _step_one_agent(self, agent):
        """Step a single agent (runs in thread pool)."""
        try:
            output = step_agent(agent, self.state, project_root=self.project_root)
            preview = output[:60].replace('\n', ' ') if output else ""
            self.events.put(RunnerEvent("step", agent.name, preview))

            if agent.is_done:
                task_id = agent.task
                branch = agent.branch
                # Complete the task (merge, cleanup, return to idle)
                try:
                    complete_task(agent, self.state, self.project_root)
                    self.events.put(RunnerEvent("done", agent.name, task_id or ""))
                    self.events.put(RunnerEvent("merged", agent.name, branch))
                except Exception as e:
                    self.events.put(RunnerEvent("error", agent.name, f"Complete failed: {e}"))

        except Exception as e:
            self.events.put(RunnerEvent("error", agent.name, str(e)))
        finally:
            with self._lock:
                self._stepping.discard(agent.name)

    def _run_loop(self):
        """Main runner loop - assigns work and steps agents in parallel."""
        import time
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=10) as executor:
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
                        executor.submit(self._step_one_agent, agent)

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

# Commands that modify state and are blocked in read-only mode
_MODIFYING_COMMANDS = frozenset({
    "spawn", "run", "stop", "kill", "cleanup", "merge",
    "reset", "new", "dep", "assign"
})


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


def cmd_stop(state, args: list[str]) -> None:
    """Stop the background runner."""
    global _runner

    if not _runner or not _runner.is_running:
        print_info("Not running")
        return

    _runner.stop()
    print_info("Stopped. Agents retain state. Use 'run' to resume.")


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
    """Show log directory for agent."""
    if not args:
        print_error("Usage: logs <name>")
        return

    name = args[0]
    log_dir = project_root / ".crew" / "logs" / name

    if log_dir.exists():
        console.print(f"[bold]Logs:[/bold] {log_dir}")
        for log_file in sorted(log_dir.glob("*.log")):
            console.print(f"  {log_file.name}")
    else:
        print_info(f"No logs for agent '{name}'")


def cmd_kill(state, args: list[str], project_root: Path) -> None:
    """Stop an agent but keep worktree."""
    if not args:
        print_error("Usage: kill <name>")
        return

    name = args[0]
    agent = state.get_agent(name)

    if not agent:
        print_error(f"Agent '{name}' not found")
        return

    agent.status = "idle"
    save_state(state, project_root)
    print_info(f"Stopped agent '{name}'. Worktree preserved at {agent.worktree}")


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
    """Show ready tickets."""
    output = run_tk("ready")
    console.print(output if output.strip() else "[dim]No ready work.[/dim]")


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


def cmd_dashboard(state, args: list[str], project_root: Path) -> None:
    """Show dashboard with runner status, costs, and agent table."""
    global _runner
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text

    # Print any pending events first
    print_runner_events()

    # Runner status
    if _runner and _runner.is_running:
        working = len([a for a in state.agents.values() if a.status in ("ready", "working")])
        idle = len([a for a in state.agents.values() if a.status == "idle"])
        if working:
            console.print(f"[bold green]● Runner active[/bold green] ({working} working, {idle} idle)")
        else:
            console.print(f"[bold green]● Runner active[/bold green] ({idle} idle)")
    else:
        console.print("[dim]○ Runner stopped[/dim]")

    # Session cost total
    total_tokens = sum(a.total_input_tokens + a.total_output_tokens for a in state.agents.values())
    total_cost = sum(a.total_cost_usd for a in state.agents.values())
    console.print(f"[bold]Session:[/bold] {total_tokens:,} tokens, ${total_cost:.4f}")
    console.print()

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

        console.print(table)

        # Log tail panels for working agents
        working_agents = [a for a in state.agents.values() if a.status in ("ready", "working")]
        if working_agents:
            console.print()
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
                    console.print(Panel(
                        truncated,
                        title=f"[bold {color}]{header}[/bold {color}]",
                        border_style=color,
                        padding=(0, 1),
                    ))

        # Git status panels for working agents
        from crew.git import run_git
        console.print()
        print_git_status_panels(list(state.agents.values()), run_git)
    else:
        console.print("[dim]No agents.[/dim]")


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

    if cmd in ("q", "quit", "exit"):
        return False
    elif cmd in ("h", "help"):
        print_help()
    elif cmd in ("dashboard", "d", "s"):
        cmd_dashboard(state, args, project_root)
    elif cmd == "spawn":
        cmd_spawn(state, args, project_root)
    elif cmd == "run":
        cmd_run(state, args, project_root)
    elif cmd == "stop":
        cmd_stop(state, args)
    elif cmd == "ps":
        cmd_ps(state, args)
    elif cmd == "peek":
        cmd_peek(state, args, project_root)
    elif cmd == "logs":
        cmd_logs(state, args, project_root)
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
    else:
        print_error(f"Unknown command: {cmd}. Type 'help' for commands.")

    return True


def main() -> None:
    """Main entry point."""
    global _read_only_mode

    project_root = Path.cwd()

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

    # Set up prompt session with history
    history_file = project_root / ".crew" / "history"
    session = PromptSession(
        history=FileHistory(str(history_file)),
        auto_suggest=AutoSuggestFromHistory(),
    )

    # Print banner
    print_banner()

    # REPL loop
    while True:
        try:
            # Print any pending events from background runner
            print_runner_events()

            line = session.prompt(get_prompt(state))
            if not handle_command(line, state, project_root):
                break
        except KeyboardInterrupt:
            # On Ctrl-C, stop the runner if running
            global _runner
            if _runner and _runner.is_running:
                _runner.stop()
                print_runner_events()
                print_info("Stopped runner")
            continue
        except EOFError:
            break

    # Stop runner on exit
    if _runner and _runner.is_running:
        _runner.stop()

    # Clean up PID file on exit (only if we're not in read-only mode)
    if not _read_only_mode:
        remove_pid_file(project_root)

    console.print("[dim]bye[/dim]")


if __name__ == "__main__":
    main()
