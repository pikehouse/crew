"""State invariant checking for crew fault injection tests.

Defines invariants that must hold true after any operation or sequence
of operations, regardless of interruptions or failures.

Invariants:
1. idle_agent_has_no_worktree - Idle agents should not have worktree assigned
2. working_agent_has_worktree - Working/ready agents must have valid worktree
3. no_duplicate_task_assignments - Each task assigned to at most one agent
4. worktree_exists_if_assigned - Worktree path exists on disk if agent refs it
5. ticket_closed_only_if_work_in_main - Ticket only closed after merge succeeds
6. session_id_valid - Non-idle agents have valid session IDs
7. branch_exists_if_working - Agent's branch exists if status is working/ready/done
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crew.state import State


@dataclass
class InvariantViolation:
    """Describes an invariant violation."""

    invariant_name: str
    agent_name: str | None
    message: str

    def __str__(self) -> str:
        if self.agent_name:
            return f"[{self.invariant_name}] Agent '{self.agent_name}': {self.message}"
        return f"[{self.invariant_name}] {self.message}"


def check_idle_agent_has_no_worktree(state: "State") -> list[InvariantViolation]:
    """Idle agents should not have a worktree assigned."""
    violations = []
    for name, agent in state.agents.items():
        if agent.status == "idle" and agent.worktree is not None:
            violations.append(InvariantViolation(
                invariant_name="idle_agent_has_no_worktree",
                agent_name=name,
                message=f"Idle agent has worktree: {agent.worktree}",
            ))
    return violations


def check_working_agent_has_worktree(state: "State") -> list[InvariantViolation]:
    """Working/ready/done agents must have a worktree assigned."""
    violations = []
    for name, agent in state.agents.items():
        if agent.status in ("working", "ready", "done") and agent.worktree is None:
            violations.append(InvariantViolation(
                invariant_name="working_agent_has_worktree",
                agent_name=name,
                message=f"Agent in '{agent.status}' status has no worktree",
            ))
    return violations


def check_no_duplicate_task_assignments(state: "State") -> list[InvariantViolation]:
    """Each task should be assigned to at most one agent."""
    violations = []
    task_to_agents: dict[str, list[str]] = {}

    for name, agent in state.agents.items():
        if agent.task:
            if agent.task not in task_to_agents:
                task_to_agents[agent.task] = []
            task_to_agents[agent.task].append(name)

    for task_id, agents in task_to_agents.items():
        if len(agents) > 1:
            violations.append(InvariantViolation(
                invariant_name="no_duplicate_task_assignments",
                agent_name=None,
                message=f"Task '{task_id}' assigned to multiple agents: {agents}",
            ))

    return violations


def check_worktree_exists_if_assigned(
    state: "State",
    project_root: Path,
) -> list[InvariantViolation]:
    """If an agent has a worktree path, it should exist on disk."""
    violations = []
    for name, agent in state.agents.items():
        if agent.worktree is not None:
            worktree_path = Path(agent.worktree)
            # Make absolute if relative
            if not worktree_path.is_absolute():
                worktree_path = project_root / worktree_path
            if not worktree_path.exists():
                violations.append(InvariantViolation(
                    invariant_name="worktree_exists_if_assigned",
                    agent_name=name,
                    message=f"Worktree path does not exist: {agent.worktree}",
                ))
    return violations


def check_ticket_closed_only_if_work_in_main(
    state: "State",
    project_root: Path,
) -> list[InvariantViolation]:
    """A ticket should only be closed if its work has been merged to main.

    This checks for the dangerous state where a ticket is closed but
    the work is not actually in main (e.g., interrupted during complete_task).

    Note: This is a heuristic check - it looks for agents in idle state
    that recently had a task, and verifies no orphaned work exists.
    """
    violations = []

    # Check all tickets in .tickets/ directory
    tickets_dir = project_root / ".tickets"
    if not tickets_dir.exists():
        return violations

    # Get list of closed tickets
    closed_tickets = set()
    for ticket_file in tickets_dir.glob("*.md"):
        try:
            content = ticket_file.read_text()
            if "status: closed" in content.lower():
                closed_tickets.add(ticket_file.stem)
        except OSError:
            continue

    # For each closed ticket, check if there's orphaned work
    for ticket_id in closed_tickets:
        # Check for orphaned branches
        branch_name = f"agent/.*-{ticket_id}"
        try:
            result = subprocess.run(
                ["git", "branch", "--list", f"*{ticket_id}*"],
                cwd=project_root,
                capture_output=True,
                text=True,
            )
            if result.stdout.strip():
                # Found branch with ticket ID - check if it's merged
                branches = result.stdout.strip().split("\n")
                for branch in branches:
                    branch = branch.strip().lstrip("* ")
                    if not branch:
                        continue
                    # Check if branch is merged into main
                    check = subprocess.run(
                        ["git", "merge-base", "--is-ancestor", branch, "main"],
                        cwd=project_root,
                        capture_output=True,
                    )
                    if check.returncode != 0:
                        violations.append(InvariantViolation(
                            invariant_name="ticket_closed_only_if_work_in_main",
                            agent_name=None,
                            message=f"Ticket '{ticket_id}' is closed but branch '{branch}' is not merged to main",
                        ))
        except Exception:
            pass

    return violations


def check_session_id_valid(state: "State") -> list[InvariantViolation]:
    """Non-idle agents should have valid session IDs."""
    violations = []
    for name, agent in state.agents.items():
        if agent.status in ("working", "ready", "done"):
            if not agent.session:
                violations.append(InvariantViolation(
                    invariant_name="session_id_valid",
                    agent_name=name,
                    message=f"Agent in '{agent.status}' status has no session ID",
                ))
    return violations


def check_branch_exists_if_working(
    state: "State",
    project_root: Path,
) -> list[InvariantViolation]:
    """Agent's branch should exist if status is working/ready/done.

    Note: Branches created in worktrees are visible from the main repo,
    but we need to check from the worktree directory if it exists.
    """
    violations = []
    for name, agent in state.agents.items():
        if agent.status in ("working", "ready", "done") and agent.branch:
            # Check from worktree if it exists, otherwise from project root
            check_dir = agent.worktree if agent.worktree and Path(agent.worktree).exists() else project_root
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "--verify", agent.branch],
                    cwd=check_dir,
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    # Also try checking from project root
                    result2 = subprocess.run(
                        ["git", "branch", "--list", agent.branch],
                        cwd=project_root,
                        capture_output=True,
                        text=True,
                    )
                    if agent.branch not in result2.stdout:
                        violations.append(InvariantViolation(
                            invariant_name="branch_exists_if_working",
                            agent_name=name,
                            message=f"Branch '{agent.branch}' does not exist",
                        ))
            except Exception as e:
                violations.append(InvariantViolation(
                    invariant_name="branch_exists_if_working",
                    agent_name=name,
                    message=f"Error checking branch '{agent.branch}': {e}",
                ))
    return violations


def check_state_file_valid(project_root: Path) -> list[InvariantViolation]:
    """State file should be valid JSON (not corrupted)."""
    violations = []
    state_file = project_root / ".crew" / "state.json"

    if state_file.exists():
        try:
            import json
            with open(state_file) as f:
                data = json.load(f)
            # Basic structure check
            if "agents" not in data:
                violations.append(InvariantViolation(
                    invariant_name="state_file_valid",
                    agent_name=None,
                    message="State file missing 'agents' key",
                ))
        except json.JSONDecodeError as e:
            violations.append(InvariantViolation(
                invariant_name="state_file_valid",
                agent_name=None,
                message=f"State file is corrupted: {e}",
            ))
        except OSError as e:
            violations.append(InvariantViolation(
                invariant_name="state_file_valid",
                agent_name=None,
                message=f"Cannot read state file: {e}",
            ))

    return violations


# All invariant checkers
INVARIANT_CHECKERS = {
    "idle_agent_has_no_worktree": check_idle_agent_has_no_worktree,
    "working_agent_has_worktree": check_working_agent_has_worktree,
    "no_duplicate_task_assignments": check_no_duplicate_task_assignments,
    "session_id_valid": check_session_id_valid,
}

# Checkers that need project_root
INVARIANT_CHECKERS_WITH_PROJECT = {
    "worktree_exists_if_assigned": check_worktree_exists_if_assigned,
    "ticket_closed_only_if_work_in_main": check_ticket_closed_only_if_work_in_main,
    "branch_exists_if_working": check_branch_exists_if_working,
}


def check_all_invariants(
    state: "State",
    project_root: Path | None = None,
    skip: list[str] | None = None,
) -> list[InvariantViolation]:
    """Check all invariants and return list of violations.

    Args:
        state: The crew state to check
        project_root: Project root for disk/git checks (optional)
        skip: List of invariant names to skip

    Returns:
        List of InvariantViolation objects (empty if all invariants hold)
    """
    skip = skip or []
    violations = []

    # Check state-only invariants
    for name, checker in INVARIANT_CHECKERS.items():
        if name not in skip:
            violations.extend(checker(state))

    # Check disk/git invariants (only if project_root provided)
    if project_root:
        for name, checker in INVARIANT_CHECKERS_WITH_PROJECT.items():
            if name not in skip:
                violations.extend(checker(state, project_root))

        # Check state file validity
        if "state_file_valid" not in skip:
            violations.extend(check_state_file_valid(project_root))

    return violations


def assert_invariants(
    state: "State",
    project_root: Path | None = None,
    skip: list[str] | None = None,
) -> None:
    """Check invariants and raise AssertionError if any are violated.

    Args:
        state: The crew state to check
        project_root: Project root for disk/git checks (optional)
        skip: List of invariant names to skip

    Raises:
        AssertionError: If any invariants are violated
    """
    violations = check_all_invariants(state, project_root, skip)
    if violations:
        msg = f"Invariant violations ({len(violations)}):\n"
        msg += "\n".join(f"  - {v}" for v in violations)
        raise AssertionError(msg)
