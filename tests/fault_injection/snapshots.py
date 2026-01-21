"""Snapshot and restore utilities for fault injection tests.

Provides the ability to capture the complete state of a crew project
(state file, worktrees, branches, tickets) and compare snapshots to
detect changes from operations.
"""

from __future__ import annotations

import json
import subprocess
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Snapshot:
    """Captures the complete state of a crew project.

    Includes:
    - State JSON (parsed)
    - Worktree directories and their file listings
    - Git branches
    - Ticket files
    """

    state_json: dict[str, Any] | None = None
    worktrees: dict[str, list[str]] = field(default_factory=dict)
    branches: list[str] = field(default_factory=list)
    tickets: dict[str, str] = field(default_factory=dict)
    git_log_main: str = ""

    @classmethod
    def capture(cls, project_root: Path) -> "Snapshot":
        """Capture a snapshot of the current project state.

        Args:
            project_root: Path to the project root

        Returns:
            Snapshot object capturing the current state
        """
        snapshot = cls()

        # Capture state.json
        state_file = project_root / ".crew" / "state.json"
        if state_file.exists():
            try:
                snapshot.state_json = json.loads(state_file.read_text())
            except (json.JSONDecodeError, OSError):
                snapshot.state_json = None

        # Capture worktrees
        agents_dir = project_root / "agents"
        if agents_dir.exists():
            for worktree_dir in agents_dir.iterdir():
                if worktree_dir.is_dir():
                    # List all files in worktree (non-recursively for speed)
                    files = []
                    try:
                        for f in worktree_dir.iterdir():
                            files.append(f.name)
                    except OSError:
                        pass
                    snapshot.worktrees[str(worktree_dir)] = sorted(files)

        # Capture git branches
        try:
            result = subprocess.run(
                ["git", "branch", "--list"],
                cwd=project_root,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                snapshot.branches = [
                    b.strip().lstrip("* ")
                    for b in result.stdout.strip().split("\n")
                    if b.strip()
                ]
        except Exception:
            pass

        # Capture tickets
        tickets_dir = project_root / ".tickets"
        if tickets_dir.exists():
            for ticket_file in tickets_dir.glob("*.md"):
                try:
                    snapshot.tickets[ticket_file.stem] = ticket_file.read_text()
                except OSError:
                    pass

        # Capture git log of main branch
        try:
            result = subprocess.run(
                ["git", "log", "main", "--oneline", "-10"],
                cwd=project_root,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                snapshot.git_log_main = result.stdout.strip()
        except Exception:
            pass

        return snapshot

    def diff(self, other: "Snapshot") -> dict[str, Any]:
        """Compare this snapshot to another and return differences.

        Args:
            other: Another snapshot to compare against

        Returns:
            Dictionary describing the differences:
            - state_changed: bool
            - worktrees_added: list of new worktree paths
            - worktrees_removed: list of removed worktree paths
            - branches_added: list of new branches
            - branches_removed: list of removed branches
            - tickets_changed: list of ticket IDs that changed
            - main_advanced: bool (True if main has new commits)
        """
        diff = {
            "state_changed": self.state_json != other.state_json,
            "worktrees_added": [],
            "worktrees_removed": [],
            "branches_added": [],
            "branches_removed": [],
            "tickets_changed": [],
            "main_advanced": self.git_log_main != other.git_log_main,
        }

        # Compare worktrees
        self_wt = set(self.worktrees.keys())
        other_wt = set(other.worktrees.keys())
        diff["worktrees_added"] = list(other_wt - self_wt)
        diff["worktrees_removed"] = list(self_wt - other_wt)

        # Compare branches
        self_branches = set(self.branches)
        other_branches = set(other.branches)
        diff["branches_added"] = list(other_branches - self_branches)
        diff["branches_removed"] = list(self_branches - other_branches)

        # Compare tickets
        all_tickets = set(self.tickets.keys()) | set(other.tickets.keys())
        for ticket_id in all_tickets:
            if self.tickets.get(ticket_id) != other.tickets.get(ticket_id):
                diff["tickets_changed"].append(ticket_id)

        return diff

    def has_agent(self, name: str) -> bool:
        """Check if state contains an agent with the given name."""
        if not self.state_json:
            return False
        agents = self.state_json.get("agents", {})
        return name in agents

    def get_agent_status(self, name: str) -> str | None:
        """Get the status of an agent, or None if not found."""
        if not self.state_json:
            return None
        agents = self.state_json.get("agents", {})
        agent = agents.get(name)
        if agent:
            return agent.get("status")
        return None

    def is_ticket_closed(self, ticket_id: str) -> bool:
        """Check if a ticket is marked as closed."""
        content = self.tickets.get(ticket_id, "")
        return "status: closed" in content.lower()

    def branch_exists(self, branch_name: str) -> bool:
        """Check if a branch exists."""
        return branch_name in self.branches


@dataclass
class ProjectFixture:
    """Helper for setting up and tearing down test projects.

    Creates a temporary git repository with:
    - Initial commit
    - Sample files
    - .tickets directory with sample tickets
    """

    root: Path
    _original_cwd: Path | None = None

    @classmethod
    def create(cls, base_dir: Path, name: str = "test-project") -> "ProjectFixture":
        """Create a new test project.

        Args:
            base_dir: Parent directory for the project
            name: Project directory name

        Returns:
            ProjectFixture instance
        """
        project_dir = base_dir / name
        project_dir.mkdir(parents=True, exist_ok=True)

        fixture = cls(root=project_dir)
        fixture._init_git()
        fixture._create_sample_files()
        fixture._create_tickets()

        return fixture

    def _init_git(self) -> None:
        """Initialize git repository."""
        subprocess.run(
            ["git", "init"],
            cwd=self.root,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=self.root,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=self.root,
            capture_output=True,
            check=True,
        )

    def _create_sample_files(self) -> None:
        """Create sample project files."""
        # Create a simple Python file
        (self.root / "main.py").write_text("# Sample project\nprint('Hello')\n")

        # Create pyproject.toml for test detection
        (self.root / "pyproject.toml").write_text(
            "[project]\nname = 'test-project'\nversion = '0.1.0'\n"
        )

        # Initial commit
        subprocess.run(
            ["git", "add", "-A"],
            cwd=self.root,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            cwd=self.root,
            capture_output=True,
            check=True,
        )

    def _create_tickets(self) -> None:
        """Create sample tickets."""
        tickets_dir = self.root / ".tickets"
        tickets_dir.mkdir(exist_ok=True)

        # Sample open ticket
        (tickets_dir / "t-0001.md").write_text(
            "---\nid: t-0001\nstatus: open\ndeps: []\n---\n# Sample task\n\nDo something.\n"
        )

        # Another open ticket
        (tickets_dir / "t-0002.md").write_text(
            "---\nid: t-0002\nstatus: open\ndeps: []\n---\n# Another task\n\nDo something else.\n"
        )

        # Commit tickets
        subprocess.run(
            ["git", "add", "-A"],
            cwd=self.root,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Add tickets"],
            cwd=self.root,
            capture_output=True,
            check=True,
        )

    def create_crew_dir(self) -> None:
        """Create the .crew directory."""
        crew_dir = self.root / ".crew"
        crew_dir.mkdir(exist_ok=True)
        (crew_dir / "logs").mkdir(exist_ok=True)

    def add_ticket(self, ticket_id: str, title: str, status: str = "open") -> None:
        """Add a new ticket."""
        tickets_dir = self.root / ".tickets"
        tickets_dir.mkdir(exist_ok=True)
        (tickets_dir / f"{ticket_id}.md").write_text(
            f"---\nid: {ticket_id}\nstatus: {status}\ndeps: []\n---\n# {title}\n\nTask description.\n"
        )

    def cleanup(self) -> None:
        """Remove the project directory."""
        if self.root.exists():
            shutil.rmtree(self.root)


def restore_snapshot(snapshot: Snapshot, project_root: Path) -> None:
    """Restore a project to a previous snapshot state.

    WARNING: This is destructive and will modify the project.
    Only use in tests with temporary directories.

    Args:
        snapshot: The snapshot to restore
        project_root: The project root to restore to
    """
    # Restore state.json
    if snapshot.state_json is not None:
        crew_dir = project_root / ".crew"
        crew_dir.mkdir(parents=True, exist_ok=True)
        state_file = crew_dir / "state.json"
        state_file.write_text(json.dumps(snapshot.state_json, indent=2))
    else:
        # Remove state file if snapshot had none
        state_file = project_root / ".crew" / "state.json"
        if state_file.exists():
            state_file.unlink()

    # Restore tickets
    tickets_dir = project_root / ".tickets"
    tickets_dir.mkdir(exist_ok=True)

    # Remove tickets not in snapshot
    for ticket_file in tickets_dir.glob("*.md"):
        if ticket_file.stem not in snapshot.tickets:
            ticket_file.unlink()

    # Write tickets from snapshot
    for ticket_id, content in snapshot.tickets.items():
        (tickets_dir / f"{ticket_id}.md").write_text(content)

    # Note: Restoring git branches and worktrees is complex and potentially
    # dangerous, so we don't do it automatically. Tests should set up
    # the git state they need explicitly.
