"""Shared pytest fixtures for crew tests."""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Generator

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
