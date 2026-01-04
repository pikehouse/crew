"""Agent runner - spawn, step, and manage Claude Code agents."""

from __future__ import annotations

import asyncio
import json
import subprocess
import uuid
from datetime import datetime
from pathlib import Path

import os
import signal

from crew.agent import Agent
from crew.crew_logging import clear_agent_logs, read_latest_log, write_log
from crew.git import create_worktree, remove_worktree, merge_branch, delete_branch, run_git
from crew.state import State, save_state


def detect_test_command(worktree: Path) -> str | None:
    """Auto-detect the appropriate test command based on project files.

    Checks for common test frameworks in this order:
    1. pytest (pyproject.toml, setup.py, pytest.ini, requirements*.txt with pytest)
    2. npm test (package.json with test script)
    3. cargo test (Cargo.toml)
    4. make test (Makefile with test target)
    5. go test (go.mod or *.go files)

    Args:
        worktree: Path to the worktree directory

    Returns:
        Test command string if detected, None if no test framework found
    """
    # Check for Python/pytest
    if (worktree / "pyproject.toml").exists():
        return "pytest"
    if (worktree / "setup.py").exists():
        return "pytest"
    if (worktree / "pytest.ini").exists():
        return "pytest"
    # Check requirements files for pytest
    for req_file in worktree.glob("requirements*.txt"):
        try:
            content = req_file.read_text()
            if "pytest" in content:
                return "pytest"
        except Exception:
            pass

    # Check for npm test
    package_json = worktree / "package.json"
    if package_json.exists():
        try:
            import json
            data = json.loads(package_json.read_text())
            if "scripts" in data and "test" in data["scripts"]:
                return "npm test"
        except Exception:
            pass

    # Check for Cargo/Rust
    if (worktree / "Cargo.toml").exists():
        return "cargo test"

    # Check for Makefile with test target
    makefile = worktree / "Makefile"
    if makefile.exists():
        try:
            content = makefile.read_text()
            # Look for test: target
            if "test:" in content or "test :" in content:
                return "make test"
        except Exception:
            pass

    # Check for Go
    if (worktree / "go.mod").exists():
        return "go test ./..."
    if list(worktree.glob("*.go")):
        return "go test ./..."

    return None


def run_tests_in_worktree(worktree: Path, timeout: int = 300) -> tuple[bool, str]:
    """Run tests in an agent's worktree.

    Auto-detects the test command and runs it.

    Args:
        worktree: Path to the worktree directory
        timeout: Timeout in seconds for test execution

    Returns:
        Tuple of (success: bool, output: str)
        - success is True if tests pass, False otherwise
        - output is the combined stdout/stderr from the test run
    """
    test_cmd = detect_test_command(worktree)

    if test_cmd is None:
        # No test command detected - consider this a pass (no tests to run)
        return True, "No test framework detected"

    try:
        result = subprocess.run(
            test_cmd,
            shell=True,
            cwd=worktree,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        output = result.stdout
        if result.stderr:
            output += "\n--- STDERR ---\n" + result.stderr

        success = result.returncode == 0
        return success, output

    except subprocess.TimeoutExpired:
        return False, f"Tests timed out after {timeout} seconds"
    except Exception as e:
        return False, f"Failed to run tests: {e}"


def format_crew_merge_message(agent: Agent) -> str:
    """Format a merge commit message with crew orchestrator attribution.

    Includes task ID, agent name, steps taken, and cost.

    Args:
        agent: The agent whose work is being merged

    Returns:
        Formatted merge commit message with crew attribution
    """
    task_id = agent.task or "no task"
    agent_name = agent.name
    steps = agent.step_count
    cost = agent.total_cost_usd

    message = f"Merge {agent.branch} ({task_id})\n\n"
    message += f"Agent: {agent_name}\n"
    message += f"Steps: {steps}\n"
    message += f"Cost: ${cost:.4f}\n\n"
    message += "🤖 Merged by crew orchestrator"

    return message


def generate_session_id() -> str:
    """Generate a new session ID (UUID)."""
    return str(uuid.uuid4())


def get_ready_tasks() -> list[str]:
    """Get list of ready task IDs from tk."""
    try:
        result = subprocess.run(
            ["tk", "ready"],
            capture_output=True,
            text=True,
        )
        tasks = []
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                parts = line.split()
                if parts:
                    tasks.append(parts[0])
        return tasks
    except FileNotFoundError:
        return []


def get_task_description(task_id: str) -> str:
    """Get task description from tk."""
    try:
        result = subprocess.run(
            ["tk", "show", task_id],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.stdout.strip() else f"Work on {task_id}"
    except FileNotFoundError:
        return f"Work on {task_id}"


# Template for agent CLAUDE.md
AGENT_TEMPLATE = """# Agent: {name}

You are an autonomous agent working in a git worktree.

## Your Task

{task_description}

## Rules

1. Stay in this directory: {worktree}
2. Commit your changes frequently with meaningful messages
3. When your task is complete, output the word DONE on its own line
4. If you discover additional work needed, create a ticket with `tk create "..."` but don't work on it yourself
5. Focus only on your assigned task

## Current Status

Task ID: {task_id}
Started: {started}

Begin working now.
"""

# Prompts
INIT_PROMPT = "Read CLAUDE.md and begin working on your assigned task."
STEP_PROMPT = "Continue working on your task. When complete, output DONE on its own line."
TEST_FAILURE_PROMPT = """Your task appeared complete, but tests failed. Please fix the failing tests.

Test output:
{test_output}

Fix the issues and when done, output DONE on its own line."""


def run_claude(
    prompt: str,
    cwd: Path,
    session: str | None = None,
    is_new_session: bool = False,
    timeout: int = 300,
) -> dict:
    """Run claude --print with JSON output and parse the response.

    Args:
        prompt: The prompt to send
        cwd: Working directory
        session: Session ID (required if is_new_session or resuming)
        is_new_session: If True, use --session-id to create new session
        timeout: Timeout in seconds

    Returns:
        Parsed JSON response with keys: result, input_tokens, output_tokens, cost_usd, stderr
    """
    cmd = ["claude", "--print", "--output-format", "json"]

    if session:
        if is_new_session:
            # First call: create session with specific ID
            cmd.extend(["--session-id", session])
        else:
            # Subsequent calls: resume existing session
            cmd.extend(["--resume", session])

    cmd.extend(["-p", prompt])

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,  # Don't wait for stdin
        )

        # Parse JSON output
        response = {
            "result": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "stderr": result.stderr,
        }

        if result.stdout.strip():
            try:
                data = json.loads(result.stdout)
                response["result"] = data.get("result", "")
                # Extract usage data
                if "usage" in data:
                    response["input_tokens"] = data["usage"].get("input_tokens", 0)
                    response["output_tokens"] = data["usage"].get("output_tokens", 0)
                response["cost_usd"] = data.get("total_cost_usd", 0.0)
            except json.JSONDecodeError:
                # If JSON parsing fails, treat stdout as plain text
                response["result"] = result.stdout

        return response

    except subprocess.TimeoutExpired:
        raise TimeoutError(f"Claude command timed out after {timeout}s")
    except FileNotFoundError:
        raise RuntimeError("Claude CLI not found. Is it installed and in PATH?")


def is_done(output: str) -> bool:
    """Check if agent output indicates task completion."""
    # Look for DONE on its own line
    lines = output.strip().split("\n")
    for line in lines:
        if line.strip().upper() == "DONE":
            return True
    return False


def spawn_worker(
    name: str,
    state: State,
    project_root: Path | None = None,
) -> Agent:
    """Spawn a new worker agent (idle, no task yet).

    Creates an agent that will wait for work to be assigned.

    Args:
        name: Agent name
        state: Current state
        project_root: Project root path

    Returns:
        The created Agent (status="idle")
    """
    project_root = project_root or Path.cwd()

    # Create agent in "idle" state (no worktree yet)
    agent = Agent(
        name=name,
        session="",  # Will be set when task is assigned
        worktree=None,
        branch="",
        task=None,
        status="idle",
        started_at=datetime.now(),
        step_count=0,
        last_step_at=None,
    )

    # Save state
    state.add_agent(agent)
    save_state(state, project_root)

    return agent


def assign_task(
    agent: Agent,
    task_id: str,
    state: State,
    project_root: Path | None = None,
) -> None:
    """Assign a task to an idle agent.

    Creates worktree, branch, and CLAUDE.md for the task.

    Args:
        agent: The agent to assign work to (must be idle)
        task_id: Ticket ID to assign
        state: Current state
        project_root: Project root path
    """
    if agent.status != "idle":
        raise RuntimeError(f"Agent {agent.name} is not idle (status={agent.status})")

    project_root = project_root or Path.cwd()
    agents_dir = project_root / "agents"

    # Clear previous task's logs so dashboard shows fresh logs for this task
    clear_agent_logs(agent.name, project_root)

    # Create unique branch name for this task
    branch_name = f"agent/{agent.name}-{task_id}"
    worktree_name = f"{agent.name}-{task_id}"

    # Create worktree
    worktree = create_worktree(worktree_name, agents_dir)

    # Get task description
    task_description = get_task_description(task_id)

    # Generate new session ID for this task
    session_id = generate_session_id()

    # Write CLAUDE.md
    claude_md = worktree / "CLAUDE.md"
    claude_md.write_text(AGENT_TEMPLATE.format(
        name=agent.name,
        task_description=task_description,
        task_id=task_id,
        worktree=worktree,
        started=datetime.now().isoformat(),
    ))

    # Update agent
    agent.session = session_id
    agent.worktree = worktree
    agent.branch = branch_name
    agent.task = task_id
    agent.status = "ready"
    agent.step_count = 0
    agent.last_step_at = None

    save_state(state, project_root)


MERGE_CONFLICT_PROMPT = """You are resolving a git merge conflict.

Multiple agents have been working on this codebase in parallel. Their changes need to be combined.

IMPORTANT: We want to KEEP BOTH sets of changes where possible. These are not conflicting features - they are complementary work from different agents that should coexist.

The conflict markers look like:
<<<<<<< HEAD
(current main branch code)
=======
(incoming branch code)
>>>>>>> branch-name

For each conflicted file:
1. Read the file to understand the conflicts
2. Edit the file to combine BOTH versions intelligently
3. Remove all conflict markers (<<<<<<, =======, >>>>>>>)
4. Make sure the result is valid, working code that includes both changes

After resolving all conflicts, run: git add -A && git commit -m "Resolve merge conflicts: combine work from multiple agents"

Then output DONE on its own line.
"""


def resolve_merge_conflicts(project_root: Path) -> bool:
    """Use Claude to resolve merge conflicts.

    Returns True if conflicts were resolved, False if resolution failed.
    """
    # Check if there are actually conflicts
    try:
        status = run_git("status", "--porcelain", cwd=project_root)
        if "UU " not in status and "AA " not in status:
            return True  # No conflicts
    except:
        pass

    # Use Claude to resolve conflicts
    session_id = generate_session_id()
    try:
        response = run_claude(
            MERGE_CONFLICT_PROMPT,
            cwd=project_root,
            session=session_id,
            is_new_session=True,
            timeout=600,  # Give it more time for complex merges
        )

        # Check if Claude said DONE
        if is_done(response["result"]):
            return True

        # Maybe it needs another step
        response2 = run_claude(
            "Continue resolving conflicts. When done, output DONE.",
            cwd=project_root,
            session=session_id,
            is_new_session=False,
            timeout=600,
        )

        return is_done(response2["result"])

    except Exception as e:
        return False


def complete_task(
    agent: Agent,
    state: State,
    project_root: Path | None = None,
) -> tuple[bool, str | None]:
    """Complete an agent's task - run tests, merge, cleanup worktree, return to idle.

    Runs tests before merging. If tests fail, returns failure status so the
    caller can feed test output back to the agent for another step.

    Args:
        agent: The agent that completed its task
        state: Current state
        project_root: Project root path

    Returns:
        Tuple of (success: bool, test_output: str | None)
        - success is True if tests passed and merge completed
        - test_output contains test failure output if tests failed, None otherwise
    """
    project_root = project_root or Path.cwd()

    # Store worktree path before any operations
    worktree_path = agent.worktree

    # Run tests FIRST (before removing worktree)
    if worktree_path:
        tests_passed, test_output = run_tests_in_worktree(worktree_path)
        if not tests_passed:
            # Tests failed - set agent back to working so it can fix the issues
            agent.status = "working"
            save_state(state, project_root)
            return False, test_output

    # Tests passed - proceed with merge

    # Close the ticket FIRST - if this fails, don't proceed with merge
    # This prevents the infinite loop where task gets re-assigned
    if agent.task:
        if not close_ticket(agent.task):
            raise RuntimeError(f"Failed to close ticket {agent.task}. Check tk status.")

    # Store branch name before removing worktree
    branch_to_merge = agent.branch

    # FIRST: Remove worktree (so the branch is no longer checked out)
    if worktree_path:
        remove_worktree(worktree_path)

    # THEN: Merge the branch from main
    run_git("checkout", "main", cwd=project_root)

    # Try the merge
    try:
        merge_branch(
            branch_to_merge,
            message=format_crew_merge_message(agent),
            cwd=project_root,
        )
    except Exception as e:
        # Merge failed - likely conflicts. Try to resolve with Claude.
        if resolve_merge_conflicts(project_root):
            # Conflicts resolved successfully
            pass
        else:
            # Claude couldn't resolve - abort and report
            try:
                run_git("merge", "--abort", cwd=project_root)
            except:
                pass
            raise RuntimeError(f"Merge failed and auto-resolution failed: {e}")

    # Delete the branch
    try:
        delete_branch(branch_to_merge, cwd=project_root)
    except:
        pass  # Branch might already be deleted

    # Commit the ticket close change (tk close modifies .tickets/xxx.md)
    # This ensures the ticket status is part of git history, not left uncommitted
    if agent.task:
        ticket_file = f".tickets/{agent.task}.md"
        try:
            run_git("add", ticket_file, cwd=project_root)
            run_git("commit", "-m", f"Close ticket {agent.task}", cwd=project_root)
        except:
            pass  # May fail if file unchanged or doesn't exist

    # Reset agent to idle (ready for next task)
    agent.session = ""
    agent.worktree = None
    agent.branch = ""
    agent.task = None
    agent.status = "idle"
    agent.step_count = 0
    agent.last_step_at = None

    save_state(state, project_root)
    return True, None


# Keep old spawn_agent for backward compatibility
def spawn_agent(
    name: str,
    task_id: str | None,
    task_description: str,
    state: State,
    project_root: Path | None = None,
) -> Agent:
    """Spawn a new agent with a task (legacy function).

    For new code, use spawn_worker() + assign_task().
    """
    project_root = project_root or Path.cwd()

    # Create worker
    agent = spawn_worker(name, state, project_root)

    # If task provided, assign it
    if task_id:
        assign_task(agent, task_id, state, project_root)

    return agent


def close_ticket(task_id: str) -> bool:
    """Close a ticket using tk.

    Returns:
        True if ticket was closed successfully, False otherwise.
    """
    import subprocess
    result = subprocess.run(["tk", "close", task_id], capture_output=True)
    return result.returncode == 0


def step_agent(
    agent: Agent,
    state: State,
    prompt: str | None = None,
    project_root: Path | None = None,
) -> str:
    """Run one step for an agent.

    Args:
        agent: The agent to step
        state: Current state
        prompt: The prompt to use (defaults based on step count)
        project_root: Project root path

    Returns:
        The Claude output (result text)
    """
    project_root = project_root or Path.cwd()

    # First step uses INIT_PROMPT, subsequent use STEP_PROMPT
    is_first_step = agent.step_count == 0
    if prompt is None:
        prompt = INIT_PROMPT if is_first_step else STEP_PROMPT

    # Set status to working
    if agent.status in ("ready", "idle"):
        agent.status = "working"

    # Run claude - first step creates session, subsequent resume it
    response = run_claude(
        prompt,
        cwd=agent.worktree,
        session=agent.session,
        is_new_session=is_first_step,
    )

    # Extract result and stderr from response
    result = response["result"]
    stderr = response["stderr"]

    # Accumulate token usage
    agent.total_input_tokens += response["input_tokens"]
    agent.total_output_tokens += response["output_tokens"]
    agent.total_cost_usd += response["cost_usd"]

    # Update step count
    agent.step_count += 1
    agent.last_step_at = datetime.now()

    # Log the interaction
    write_log(
        agent_name=agent.name,
        log_type="step",
        prompt=prompt,
        output=result + ("\n---STDERR---\n" + stderr if stderr else ""),
        session=agent.session,
        step=agent.step_count,
        project_root=project_root,
    )

    # Check if done (but don't close ticket yet - that happens after merge)
    if is_done(result):
        agent.status = "done"

    # Check if stuck (too many steps)
    if agent.step_count >= 20 and agent.status != "done":
        agent.status = "stuck"

    # Save state
    save_state(state, project_root)

    return result


def cleanup_agent(
    agent: Agent,
    state: State,
    merge: bool = True,
    project_root: Path | None = None,
) -> None:
    """Clean up an agent - optionally merge and remove worktree.

    Args:
        agent: The agent to clean up
        state: Current state
        merge: Whether to merge the branch
        project_root: Project root path
    """
    project_root = project_root or Path.cwd()

    if merge:
        # Switch to main and merge
        run_git("checkout", "main", cwd=project_root)
        try:
            merge_branch(
                agent.branch,
                message=format_crew_merge_message(agent),
                cwd=project_root,
            )
            delete_branch(agent.branch, cwd=project_root)
        except Exception as e:
            raise RuntimeError(f"Merge failed: {e}. Resolve conflicts manually.")

    # Remove worktree
    remove_worktree(agent.worktree)

    # Remove from state
    state.remove_agent(agent.name)
    save_state(state, project_root)


def find_claude_process(agent: Agent) -> int | None:
    """Find Claude process for an agent by session ID or worktree path.

    Returns:
        Process ID if found, None otherwise.
    """
    try:
        # Get all running processes with full command line
        result = subprocess.run(
            ["ps", "-eo", "pid,command"],
            capture_output=True,
            text=True,
        )

        for line in result.stdout.strip().split("\n")[1:]:  # Skip header
            if "claude" not in line.lower():
                continue
            if "--print" not in line:
                continue

            parts = line.strip().split(None, 1)
            if len(parts) < 2:
                continue

            pid_str, cmd = parts

            # Match by session ID
            if agent.session and agent.session in cmd:
                return int(pid_str)

            # Match by worktree path
            if agent.worktree and str(agent.worktree) in cmd:
                return int(pid_str)

        return None
    except Exception:
        return None


def kill_claude_process(agent: Agent) -> bool:
    """Kill the Claude process for an agent.

    Returns:
        True if process was found and killed, False otherwise.
    """
    pid = find_claude_process(agent)
    if pid is None:
        return False

    try:
        os.kill(pid, signal.SIGTERM)
        return True
    except ProcessLookupError:
        # Process already gone
        return False
    except PermissionError:
        return False


def check_work_completed(agent: Agent, project_root: Path | None = None) -> str:
    """Check if agent's work was completed by parsing logs.

    Returns:
        "done" - DONE marker found in logs (work completed)
        "partial" - Some steps taken but not completed
        "nothing" - No work done (no logs or empty logs)
    """
    project_root = project_root or Path.cwd()

    # Read the latest log
    log_content = read_latest_log(agent.name, project_root)

    if not log_content:
        return "nothing"

    # Check if DONE is in the log output
    # Extract the output section (between --- and === END ===)
    parts = log_content.split("---\n", 1)
    if len(parts) < 2:
        output = log_content
    else:
        output = parts[1].rsplit("=== END ===", 1)[0]

    if is_done(output):
        return "done"

    # If there's meaningful output, consider it partial work
    if agent.step_count > 0:
        return "partial"

    return "nothing"


def shutdown_agent(
    agent: Agent,
    state: State,
    project_root: Path | None = None,
) -> str:
    """Shutdown an agent gracefully, reconciling its state.

    This function:
    1. Finds and kills the Claude process by session ID or worktree path
    2. Checks if work was completed (parses logs for DONE)
    3. Reconciles state based on work status:
       - done: Set status to "done" (ready for merge)
       - partial: Keep status as "working" (can resume later)
       - nothing: Reset to "ready" (start fresh)
    4. Saves state

    Args:
        agent: The agent to shutdown
        state: Current state
        project_root: Project root path

    Returns:
        Work completion status: "done", "partial", or "nothing"
    """
    project_root = project_root or Path.cwd()

    # Step 1: Kill the Claude process
    process_killed = kill_claude_process(agent)

    # Step 2: Check work completion status
    work_status = check_work_completed(agent, project_root)

    # Step 3: Reconcile state based on work status
    if work_status == "done":
        # Work completed - mark as done (ready for merge via complete_task)
        agent.status = "done"
    elif work_status == "partial":
        # Partial work - keep in working state so it can be resumed
        # Status remains "working" (or set to working if it was something else)
        if agent.status not in ("done", "stuck"):
            agent.status = "working"
    else:
        # No work done - reset to ready state
        agent.status = "ready"

    # Step 4: Save state
    save_state(state, project_root)

    return work_status


async def step_agent_async(
    agent: Agent,
    state: State,
    project_root: Path | None = None,
) -> str:
    """Async wrapper for step_agent."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: step_agent(agent, state, project_root=project_root),
    )


async def run_agents(
    state: State,
    project_root: Path | None = None,
    on_step: callable = None,
    on_done: callable = None,
) -> None:
    """Continuously step all active agents until all done.

    Args:
        state: Current state
        project_root: Project root path
        on_step: Callback(agent, output) after each step
        on_done: Callback(agent) when agent completes
    """
    project_root = project_root or Path.cwd()

    while state.active_agents:
        # Step all active agents in parallel
        tasks = [
            step_agent_async(agent, state, project_root)
            for agent in state.active_agents
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for agent, result in zip(list(state.active_agents), results):
            if isinstance(result, Exception):
                agent.status = "stuck"
                save_state(state, project_root)
                continue

            if on_step:
                on_step(agent, result)

            if agent.is_done:
                if on_done:
                    on_done(agent)

                # Auto-merge
                try:
                    cleanup_agent(agent, state, merge=True, project_root=project_root)
                except Exception as e:
                    if on_step:
                        on_step(agent, f"Merge failed: {e}")

        # Brief pause between rounds
        await asyncio.sleep(1)
