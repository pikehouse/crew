"""Shared pytest fixtures for crew tests."""

from __future__ import annotations

import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Generator
from unittest.mock import patch

import pytest

from crew.agent import Agent
from crew.state import State, ensure_crew_dir, save_state


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test isolation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def project_root(temp_dir: Path) -> Path:
    """Create a mock project root with .crew directory."""
    ensure_crew_dir(temp_dir)
    return temp_dir


@pytest.fixture
def sample_agent() -> Agent:
    """Create a sample agent for testing."""
    return Agent(
        name="test-agent",
        session="test-session-123",
        worktree=Path("/tmp/test-worktree"),
        branch="agent/test-branch",
        task="Test task description",
        status="idle",
        started_at=datetime(2026, 1, 1, 12, 0, 0),
        step_count=0,
        last_step_at=None,
    )


@pytest.fixture
def working_agent() -> Agent:
    """Create an agent in working state."""
    return Agent(
        name="working-agent",
        session="working-session-456",
        worktree=Path("/tmp/working-worktree"),
        branch="agent/working-branch",
        task="Working on something",
        status="working",
        started_at=datetime(2026, 1, 1, 10, 0, 0),
        step_count=5,
        last_step_at=datetime(2026, 1, 1, 11, 30, 0),
    )


@pytest.fixture
def empty_state() -> State:
    """Create an empty state for testing."""
    return State()


@pytest.fixture
def populated_state(sample_agent: Agent, working_agent: Agent) -> State:
    """Create a state with multiple agents."""
    state = State()
    state.add_agent(sample_agent)
    state.add_agent(working_agent)
    return state


@pytest.fixture
def state_with_file(project_root: Path, populated_state: State) -> tuple[Path, State]:
    """Create a state and save it to disk."""
    save_state(populated_state, project_root)
    return project_root, populated_state


# =============================================================================
# Fault Injection Test Fixtures
# =============================================================================


@pytest.fixture
def git_project(temp_dir: Path) -> Generator[Path, None, None]:
    """Create a temporary directory initialized as a git repository.

    Includes:
    - Git repo with initial commit
    - Sample files (main.py, pyproject.toml)
    - .tickets directory with sample tickets
    - .crew directory created
    """
    # Initialize git
    subprocess.run(
        ["git", "init"],
        cwd=temp_dir,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=temp_dir,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=temp_dir,
        capture_output=True,
        check=True,
    )

    # Create sample files
    (temp_dir / "main.py").write_text("# Sample project\nprint('Hello')\n")
    (temp_dir / "pyproject.toml").write_text(
        "[project]\nname = 'test-project'\nversion = '0.1.0'\n"
    )

    # Create .tickets directory with sample tickets
    tickets_dir = temp_dir / ".tickets"
    tickets_dir.mkdir(exist_ok=True)
    (tickets_dir / "t-0001.md").write_text(
        "---\nid: t-0001\nstatus: open\ndeps: []\n---\n# Sample task\n\nDo something.\n"
    )
    (tickets_dir / "t-0002.md").write_text(
        "---\nid: t-0002\nstatus: open\ndeps: []\n---\n# Another task\n\nDo something else.\n"
    )

    # Initial commit
    subprocess.run(
        ["git", "add", "-A"],
        cwd=temp_dir,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=temp_dir,
        capture_output=True,
        check=True,
    )

    # Create agents directory
    (temp_dir / "agents").mkdir(exist_ok=True)

    # Create .crew directory
    ensure_crew_dir(temp_dir)

    yield temp_dir


@pytest.fixture
def mock_claude():
    """Provide a configurable MockClaude for testing.

    Returns the MockClaude class for configuration in tests.
    """
    from tests.fixtures.mock_claude import MockClaude

    return MockClaude()


@pytest.fixture
def mock_tk():
    """Mock the tk command for ticket operations."""

    def _mock_run(*args, **kwargs):
        """Mock subprocess.run for tk commands."""
        cmd = args[0] if args else kwargs.get("command", [])

        class MockResult:
            returncode = 0
            stdout = ""
            stderr = ""

        result = MockResult()

        if isinstance(cmd, list) and len(cmd) >= 2:
            if cmd[0] == "tk":
                if cmd[1] == "ready":
                    result.stdout = "t-0001\nt-0002\n"
                elif cmd[1] == "show" and len(cmd) >= 3:
                    task_id = cmd[2]
                    result.stdout = f"---\nid: {task_id}\nstatus: open\n---\n# Task {task_id}\n\nWork on this task.\n"
                elif cmd[1] == "close":
                    result.returncode = 0

        return result

    with patch("subprocess.run", side_effect=_mock_run):
        yield


@pytest.fixture
def idle_agent_in_state(git_project: Path) -> tuple[Agent, State, Path]:
    """Create an idle agent saved to state in a git project.

    Returns:
        Tuple of (agent, state, project_root)
    """
    state = State()
    agent = Agent(
        name="test-agent",
        session="",
        worktree=None,
        branch="",
        task=None,
        status="idle",
        started_at=datetime.now(),
        step_count=0,
        last_step_at=None,
    )
    state.add_agent(agent)
    save_state(state, git_project)
    return agent, state, git_project


@pytest.fixture
def ready_agent_in_state(git_project: Path) -> tuple[Agent, State, Path]:
    """Create a ready agent with worktree in a git project.

    Returns:
        Tuple of (agent, state, project_root)
    """
    from crew.runner import spawn_worker, assign_task

    state = State()

    # Create idle agent
    agent = spawn_worker("test-agent", state, git_project)

    # Mock tk commands for assign_task
    def mock_run(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("command", [])
        class MockResult:
            returncode = 0
            stdout = ""
            stderr = ""
        result = MockResult()

        if isinstance(cmd, list):
            if cmd[0] == "tk":
                if cmd[1] == "show":
                    result.stdout = "---\nid: t-0001\nstatus: open\n---\n# Task\n\nDo something.\n"
            elif cmd[0] == "git":
                # Let git commands through
                return subprocess.run(*args, **kwargs)

        return result

    # Assign task
    with patch("crew.runner.subprocess.run", side_effect=mock_run):
        assign_task(agent, "t-0001", state, git_project)

    return agent, state, git_project
