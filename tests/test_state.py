"""Tests for crew.state module."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from crew.agent import Agent
from crew.state import (
    CREW_DIR,
    STATE_FILE,
    STATE_VERSION,
    State,
    ensure_crew_dir,
    get_crew_dir,
    load_state,
    save_state,
)


class TestStateDefaults:
    """Test State dataclass defaults and initialization."""

    def test_empty_state_has_no_agents(self, empty_state: State):
        """Test that empty state has no agents."""
        assert empty_state.agents == {}

    def test_empty_state_has_current_version(self, empty_state: State):
        """Test that empty state has current version."""
        assert empty_state.version == STATE_VERSION

    def test_state_version_constant(self):
        """Test that STATE_VERSION is set correctly."""
        assert STATE_VERSION == 1


class TestStateAgentManagement:
    """Test State agent management methods."""

    def test_add_agent(self, empty_state: State, sample_agent: Agent):
        """Test adding an agent to state."""
        empty_state.add_agent(sample_agent)
        assert sample_agent.name in empty_state.agents
        assert empty_state.agents[sample_agent.name] == sample_agent

    def test_add_agent_overwrites_existing(self, empty_state: State):
        """Test that adding an agent with same name overwrites."""
        agent1 = Agent(
            name="test",
            session="sess-1",
            worktree=Path("/tmp/1"),
            branch="branch1",
        )
        agent2 = Agent(
            name="test",
            session="sess-2",
            worktree=Path("/tmp/2"),
            branch="branch2",
        )

        empty_state.add_agent(agent1)
        empty_state.add_agent(agent2)

        assert len(empty_state.agents) == 1
        assert empty_state.agents["test"].session == "sess-2"

    def test_get_agent_returns_agent(self, populated_state: State, sample_agent: Agent):
        """Test getting an existing agent."""
        agent = populated_state.get_agent(sample_agent.name)
        assert agent is not None
        assert agent.name == sample_agent.name

    def test_get_agent_returns_none_for_missing(self, populated_state: State):
        """Test getting a non-existent agent returns None."""
        agent = populated_state.get_agent("non-existent")
        assert agent is None

    def test_remove_agent(self, populated_state: State, sample_agent: Agent):
        """Test removing an agent."""
        populated_state.remove_agent(sample_agent.name)
        assert sample_agent.name not in populated_state.agents

    def test_remove_nonexistent_agent_is_safe(self, empty_state: State):
        """Test that removing a non-existent agent doesn't raise."""
        empty_state.remove_agent("non-existent")  # Should not raise


class TestStateActiveAgents:
    """Test State active_agents property."""

    def test_active_agents_returns_active_only(self, populated_state: State):
        """Test that active_agents returns only agents with active status."""
        active = populated_state.active_agents
        for agent in active:
            assert agent.is_active

    def test_active_agents_excludes_done(self, empty_state: State):
        """Test that done agents are excluded from active_agents."""
        done_agent = Agent(
            name="done",
            session="sess-done",
            worktree=Path("/tmp/done"),
            branch="main",
            status="done",
        )
        empty_state.add_agent(done_agent)

        assert len(empty_state.active_agents) == 0

    def test_active_agents_excludes_stuck(self, empty_state: State):
        """Test that stuck agents are excluded from active_agents."""
        stuck_agent = Agent(
            name="stuck",
            session="sess-stuck",
            worktree=Path("/tmp/stuck"),
            branch="main",
            status="stuck",
        )
        empty_state.add_agent(stuck_agent)

        assert len(empty_state.active_agents) == 0

    def test_active_agents_includes_idle_ready_working(self, empty_state: State):
        """Test that idle, ready, and working agents are included."""
        for i, status in enumerate(["idle", "ready", "working"]):
            agent = Agent(
                name=f"agent-{status}",
                session=f"sess-{i}",
                worktree=Path(f"/tmp/{status}"),
                branch="main",
                status=status,
            )
            empty_state.add_agent(agent)

        active = empty_state.active_agents
        assert len(active) == 3


class TestStateSerialization:
    """Test State serialization and deserialization."""

    def test_to_dict_structure(self, populated_state: State):
        """Test that to_dict returns correct structure."""
        data = populated_state.to_dict()

        assert "agents" in data
        assert "version" in data
        assert isinstance(data["agents"], dict)
        assert data["version"] == STATE_VERSION

    def test_to_dict_serializes_all_agents(self, populated_state: State):
        """Test that to_dict serializes all agents."""
        data = populated_state.to_dict()

        assert len(data["agents"]) == len(populated_state.agents)
        for name in populated_state.agents:
            assert name in data["agents"]

    def test_to_dict_agents_are_dicts(self, populated_state: State):
        """Test that agents in to_dict output are dicts."""
        data = populated_state.to_dict()

        for agent_data in data["agents"].values():
            assert isinstance(agent_data, dict)
            assert "name" in agent_data
            assert "session" in agent_data

    def test_from_dict_restores_state(self):
        """Test that from_dict restores state correctly."""
        data = {
            "agents": {
                "agent-1": {
                    "name": "agent-1",
                    "session": "sess-1",
                    "worktree": "/tmp/1",
                    "branch": "main",
                    "started_at": "2026-01-01T12:00:00",
                    "total_input_tokens": 1000,
                    "total_output_tokens": 500,
                    "total_cost_usd": 0.05,
                },
            },
            "version": 1,
        }

        state = State.from_dict(data)

        assert len(state.agents) == 1
        assert "agent-1" in state.agents
        assert state.version == 1

    def test_from_dict_restores_token_fields(self):
        """Test that from_dict restores token tracking fields."""
        data = {
            "agents": {
                "token-agent": {
                    "name": "token-agent",
                    "session": "sess-1",
                    "worktree": "/tmp/tokens",
                    "branch": "main",
                    "started_at": "2026-01-01T12:00:00",
                    "total_input_tokens": 5000,
                    "total_output_tokens": 2500,
                    "total_cost_usd": 0.25,
                },
            },
            "version": 1,
        }

        state = State.from_dict(data)
        agent = state.get_agent("token-agent")

        assert agent is not None
        assert agent.total_input_tokens == 5000
        assert agent.total_output_tokens == 2500
        assert agent.total_cost_usd == 0.25

    def test_from_dict_with_empty_agents(self):
        """Test from_dict with empty agents dict."""
        data = {"agents": {}, "version": 1}

        state = State.from_dict(data)

        assert len(state.agents) == 0
        assert state.version == 1

    def test_from_dict_with_missing_agents_key(self):
        """Test from_dict handles missing agents key."""
        data = {"version": 1}

        state = State.from_dict(data)

        assert len(state.agents) == 0

    def test_roundtrip_serialization(self, populated_state: State):
        """Test that to_dict and from_dict are inverse operations."""
        data = populated_state.to_dict()
        restored = State.from_dict(data)

        assert len(restored.agents) == len(populated_state.agents)
        assert restored.version == populated_state.version

        for name, original_agent in populated_state.agents.items():
            restored_agent = restored.get_agent(name)
            assert restored_agent is not None
            assert restored_agent.name == original_agent.name
            assert restored_agent.session == original_agent.session

    def test_roundtrip_preserves_token_data(self, empty_state: State):
        """Test that roundtrip serialization preserves token data."""
        agent = Agent(
            name="token-test",
            session="sess-tokens",
            worktree=Path("/tmp/tokens"),
            branch="main",
            total_input_tokens=10000,
            total_output_tokens=5000,
            total_cost_usd=0.50,
        )
        empty_state.add_agent(agent)

        data = empty_state.to_dict()
        restored = State.from_dict(data)
        restored_agent = restored.get_agent("token-test")

        assert restored_agent is not None
        assert restored_agent.total_input_tokens == 10000
        assert restored_agent.total_output_tokens == 5000
        assert restored_agent.total_cost_usd == 0.50


class TestCrewDirFunctions:
    """Test crew directory utility functions."""

    def test_get_crew_dir_returns_crew_subdir(self, temp_dir: Path):
        """Test get_crew_dir returns .crew subdirectory."""
        crew_dir = get_crew_dir(temp_dir)
        assert crew_dir == temp_dir / CREW_DIR

    def test_get_crew_dir_uses_cwd_by_default(self):
        """Test get_crew_dir uses current directory by default."""
        crew_dir = get_crew_dir()
        assert crew_dir == Path.cwd() / CREW_DIR

    def test_ensure_crew_dir_creates_directory(self, temp_dir: Path):
        """Test ensure_crew_dir creates .crew directory."""
        crew_dir = ensure_crew_dir(temp_dir)

        assert crew_dir.exists()
        assert crew_dir.is_dir()
        assert crew_dir == temp_dir / CREW_DIR

    def test_ensure_crew_dir_creates_logs_subdir(self, temp_dir: Path):
        """Test ensure_crew_dir creates logs subdirectory."""
        ensure_crew_dir(temp_dir)

        logs_dir = temp_dir / CREW_DIR / "logs"
        assert logs_dir.exists()
        assert logs_dir.is_dir()

    def test_ensure_crew_dir_is_idempotent(self, temp_dir: Path):
        """Test ensure_crew_dir can be called multiple times safely."""
        ensure_crew_dir(temp_dir)
        ensure_crew_dir(temp_dir)  # Should not raise

        assert (temp_dir / CREW_DIR).exists()


class TestSaveLoadState:
    """Test state persistence functions."""

    def test_save_state_creates_file(self, project_root: Path, empty_state: State):
        """Test save_state creates state.json file."""
        save_state(empty_state, project_root)

        state_file = project_root / CREW_DIR / STATE_FILE
        assert state_file.exists()

    def test_save_state_writes_valid_json(self, project_root: Path, populated_state: State):
        """Test save_state writes valid JSON."""
        save_state(populated_state, project_root)

        state_file = project_root / CREW_DIR / STATE_FILE
        with open(state_file) as f:
            data = json.load(f)  # Should not raise

        assert "agents" in data
        assert "version" in data

    def test_save_state_creates_crew_dir_if_missing(self, temp_dir: Path, empty_state: State):
        """Test save_state creates .crew directory if it doesn't exist."""
        save_state(empty_state, temp_dir)

        assert (temp_dir / CREW_DIR).exists()

    def test_load_state_returns_empty_when_no_file(self, project_root: Path):
        """Test load_state returns empty state when file doesn't exist."""
        state = load_state(project_root)

        assert len(state.agents) == 0
        assert state.version == STATE_VERSION

    def test_load_state_reads_saved_state(self, state_with_file: tuple[Path, State]):
        """Test load_state reads previously saved state."""
        project_root, original_state = state_with_file

        loaded_state = load_state(project_root)

        assert len(loaded_state.agents) == len(original_state.agents)

    def test_save_load_roundtrip(self, project_root: Path, populated_state: State):
        """Test save and load are inverse operations."""
        save_state(populated_state, project_root)
        loaded_state = load_state(project_root)

        assert len(loaded_state.agents) == len(populated_state.agents)
        assert loaded_state.version == populated_state.version

        for name in populated_state.agents:
            assert name in loaded_state.agents

    def test_save_load_preserves_token_data(self, project_root: Path, empty_state: State):
        """Test save/load roundtrip preserves token tracking data."""
        agent = Agent(
            name="token-persist",
            session="sess-persist",
            worktree=Path("/tmp/persist"),
            branch="main",
            total_input_tokens=15000,
            total_output_tokens=7500,
            total_cost_usd=0.75,
        )
        empty_state.add_agent(agent)

        save_state(empty_state, project_root)
        loaded_state = load_state(project_root)
        loaded_agent = loaded_state.get_agent("token-persist")

        assert loaded_agent is not None
        assert loaded_agent.total_input_tokens == 15000
        assert loaded_agent.total_output_tokens == 7500
        assert loaded_agent.total_cost_usd == 0.75

    def test_save_overwrites_existing_file(self, project_root: Path):
        """Test save_state overwrites existing state file."""
        # Save initial state
        state1 = State()
        agent1 = Agent(
            name="agent1",
            session="sess-1",
            worktree=Path("/tmp/1"),
            branch="main",
        )
        state1.add_agent(agent1)
        save_state(state1, project_root)

        # Save different state
        state2 = State()
        agent2 = Agent(
            name="agent2",
            session="sess-2",
            worktree=Path("/tmp/2"),
            branch="main",
        )
        state2.add_agent(agent2)
        save_state(state2, project_root)

        # Load and verify
        loaded = load_state(project_root)
        assert "agent1" not in loaded.agents
        assert "agent2" in loaded.agents
