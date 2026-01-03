"""Tests for crew.agent module."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from crew.agent import Agent, AgentStatus


class TestAgentDefaults:
    """Test Agent dataclass defaults and initialization."""

    def test_minimal_creation(self):
        """Test creating agent with minimal required fields."""
        agent = Agent(
            name="test",
            session="sess-123",
            worktree=Path("/tmp/test"),
            branch="main",
        )
        assert agent.name == "test"
        assert agent.session == "sess-123"
        assert agent.worktree == Path("/tmp/test")
        assert agent.branch == "main"

    def test_default_status_is_idle(self):
        """Test that default status is 'idle'."""
        agent = Agent(
            name="test",
            session="sess-123",
            worktree=Path("/tmp/test"),
            branch="main",
        )
        assert agent.status == "idle"

    def test_default_task_is_none(self):
        """Test that default task is None."""
        agent = Agent(
            name="test",
            session="sess-123",
            worktree=Path("/tmp/test"),
            branch="main",
        )
        assert agent.task is None

    def test_default_step_count_is_zero(self):
        """Test that default step_count is 0."""
        agent = Agent(
            name="test",
            session="sess-123",
            worktree=Path("/tmp/test"),
            branch="main",
        )
        assert agent.step_count == 0

    def test_default_last_step_at_is_none(self):
        """Test that default last_step_at is None."""
        agent = Agent(
            name="test",
            session="sess-123",
            worktree=Path("/tmp/test"),
            branch="main",
        )
        assert agent.last_step_at is None

    def test_started_at_defaults_to_now(self):
        """Test that started_at defaults to approximately now."""
        before = datetime.now()
        agent = Agent(
            name="test",
            session="sess-123",
            worktree=Path("/tmp/test"),
            branch="main",
        )
        after = datetime.now()
        assert before <= agent.started_at <= after

    def test_worktree_can_be_none(self):
        """Test that worktree can be None."""
        agent = Agent(
            name="test",
            session="sess-123",
            worktree=None,
            branch="main",
        )
        assert agent.worktree is None


class TestAgentTokenFieldDefaults:
    """Test token tracking field defaults."""

    def test_default_total_input_tokens_is_zero(self):
        """Test that default total_input_tokens is 0."""
        agent = Agent(
            name="test",
            session="sess-123",
            worktree=Path("/tmp/test"),
            branch="main",
        )
        assert agent.total_input_tokens == 0

    def test_default_total_output_tokens_is_zero(self):
        """Test that default total_output_tokens is 0."""
        agent = Agent(
            name="test",
            session="sess-123",
            worktree=Path("/tmp/test"),
            branch="main",
        )
        assert agent.total_output_tokens == 0

    def test_default_total_cost_usd_is_zero(self):
        """Test that default total_cost_usd is 0.0."""
        agent = Agent(
            name="test",
            session="sess-123",
            worktree=Path("/tmp/test"),
            branch="main",
        )
        assert agent.total_cost_usd == 0.0

    def test_token_fields_can_be_set(self):
        """Test that token fields can be set during creation."""
        agent = Agent(
            name="test",
            session="sess-123",
            worktree=Path("/tmp/test"),
            branch="main",
            total_input_tokens=1000,
            total_output_tokens=500,
            total_cost_usd=0.05,
        )
        assert agent.total_input_tokens == 1000
        assert agent.total_output_tokens == 500
        assert agent.total_cost_usd == 0.05


class TestAgentStateTransitions:
    """Test Agent status-related properties and transitions."""

    @pytest.mark.parametrize("status", ["idle", "ready", "working"])
    def test_is_active_for_active_statuses(self, status: AgentStatus):
        """Test is_active returns True for active statuses."""
        agent = Agent(
            name="test",
            session="sess-123",
            worktree=Path("/tmp/test"),
            branch="main",
            status=status,
        )
        assert agent.is_active is True

    @pytest.mark.parametrize("status", ["done", "stuck"])
    def test_is_active_for_inactive_statuses(self, status: AgentStatus):
        """Test is_active returns False for inactive statuses."""
        agent = Agent(
            name="test",
            session="sess-123",
            worktree=Path("/tmp/test"),
            branch="main",
            status=status,
        )
        assert agent.is_active is False

    def test_is_done_when_status_is_done(self):
        """Test is_done returns True when status is 'done'."""
        agent = Agent(
            name="test",
            session="sess-123",
            worktree=Path("/tmp/test"),
            branch="main",
            status="done",
        )
        assert agent.is_done is True

    @pytest.mark.parametrize("status", ["idle", "ready", "working", "stuck"])
    def test_is_done_when_status_is_not_done(self, status: AgentStatus):
        """Test is_done returns False when status is not 'done'."""
        agent = Agent(
            name="test",
            session="sess-123",
            worktree=Path("/tmp/test"),
            branch="main",
            status=status,
        )
        assert agent.is_done is False

    def test_status_can_be_changed(self):
        """Test that agent status can be modified."""
        agent = Agent(
            name="test",
            session="sess-123",
            worktree=Path("/tmp/test"),
            branch="main",
            status="idle",
        )
        assert agent.status == "idle"

        agent.status = "working"
        assert agent.status == "working"
        assert agent.is_active is True

        agent.status = "done"
        assert agent.status == "done"
        assert agent.is_done is True
        assert agent.is_active is False


class TestAgentSerialization:
    """Test Agent serialization and deserialization."""

    def test_to_dict_includes_all_fields(self, sample_agent: Agent):
        """Test that to_dict includes all fields."""
        data = sample_agent.to_dict()

        assert data["name"] == "test-agent"
        assert data["session"] == "test-session-123"
        assert data["worktree"] == "/tmp/test-worktree"
        assert data["branch"] == "agent/test-branch"
        assert data["task"] == "Test task description"
        assert data["status"] == "idle"
        assert data["started_at"] == "2026-01-01T12:00:00"
        assert data["step_count"] == 0
        assert data["last_step_at"] is None

    def test_to_dict_includes_token_fields(self):
        """Test that to_dict includes token tracking fields."""
        agent = Agent(
            name="test",
            session="sess-123",
            worktree=Path("/tmp/test"),
            branch="main",
            total_input_tokens=1500,
            total_output_tokens=750,
            total_cost_usd=0.075,
        )
        data = agent.to_dict()

        assert data["total_input_tokens"] == 1500
        assert data["total_output_tokens"] == 750
        assert data["total_cost_usd"] == 0.075

    def test_to_dict_with_none_worktree(self):
        """Test to_dict when worktree is None."""
        agent = Agent(
            name="test",
            session="sess-123",
            worktree=None,
            branch="main",
        )
        data = agent.to_dict()
        assert data["worktree"] is None

    def test_to_dict_with_last_step_at(self, working_agent: Agent):
        """Test to_dict includes last_step_at when set."""
        data = working_agent.to_dict()
        assert data["last_step_at"] == "2026-01-01T11:30:00"

    def test_from_dict_restores_all_fields(self):
        """Test from_dict restores all fields correctly."""
        data = {
            "name": "restored-agent",
            "session": "restored-session",
            "worktree": "/tmp/restored",
            "branch": "feature/test",
            "task": "Restored task",
            "status": "working",
            "started_at": "2026-01-02T14:30:00",
            "step_count": 10,
            "last_step_at": "2026-01-02T15:00:00",
            "total_input_tokens": 2000,
            "total_output_tokens": 1000,
            "total_cost_usd": 0.10,
        }

        agent = Agent.from_dict(data)

        assert agent.name == "restored-agent"
        assert agent.session == "restored-session"
        assert agent.worktree == Path("/tmp/restored")
        assert agent.branch == "feature/test"
        assert agent.task == "Restored task"
        assert agent.status == "working"
        assert agent.started_at == datetime(2026, 1, 2, 14, 30, 0)
        assert agent.step_count == 10
        assert agent.last_step_at == datetime(2026, 1, 2, 15, 0, 0)
        assert agent.total_input_tokens == 2000
        assert agent.total_output_tokens == 1000
        assert agent.total_cost_usd == 0.10

    def test_from_dict_with_none_worktree(self):
        """Test from_dict handles None worktree."""
        data = {
            "name": "test",
            "session": "sess-123",
            "worktree": None,
            "branch": "main",
            "started_at": "2026-01-01T12:00:00",
        }
        agent = Agent.from_dict(data)
        assert agent.worktree is None

    def test_from_dict_with_missing_optional_fields(self):
        """Test from_dict handles missing optional fields with defaults."""
        data = {
            "name": "minimal",
            "session": "sess-123",
            "branch": "main",
            "started_at": "2026-01-01T12:00:00",
        }
        agent = Agent.from_dict(data)

        assert agent.worktree is None
        assert agent.task is None
        assert agent.status == "idle"
        assert agent.step_count == 0
        assert agent.last_step_at is None
        assert agent.total_input_tokens == 0
        assert agent.total_output_tokens == 0
        assert agent.total_cost_usd == 0.0

    def test_roundtrip_serialization(self, sample_agent: Agent):
        """Test that to_dict and from_dict are inverse operations."""
        data = sample_agent.to_dict()
        restored = Agent.from_dict(data)

        assert restored.name == sample_agent.name
        assert restored.session == sample_agent.session
        assert restored.worktree == sample_agent.worktree
        assert restored.branch == sample_agent.branch
        assert restored.task == sample_agent.task
        assert restored.status == sample_agent.status
        assert restored.started_at == sample_agent.started_at
        assert restored.step_count == sample_agent.step_count
        assert restored.last_step_at == sample_agent.last_step_at

    def test_roundtrip_with_token_fields(self):
        """Test roundtrip serialization preserves token fields."""
        original = Agent(
            name="token-test",
            session="sess-123",
            worktree=Path("/tmp/test"),
            branch="main",
            total_input_tokens=5000,
            total_output_tokens=2500,
            total_cost_usd=0.25,
        )
        data = original.to_dict()
        restored = Agent.from_dict(data)

        assert restored.total_input_tokens == original.total_input_tokens
        assert restored.total_output_tokens == original.total_output_tokens
        assert restored.total_cost_usd == original.total_cost_usd


class TestAgentElapsed:
    """Test Agent elapsed time calculation."""

    def test_elapsed_returns_minutes_for_short_duration(self):
        """Test elapsed returns minutes for durations under an hour."""
        agent = Agent(
            name="test",
            session="sess-123",
            worktree=Path("/tmp/test"),
            branch="main",
            started_at=datetime.now() - timedelta(minutes=30),
        )
        elapsed = agent.elapsed
        # Should be approximately "30m" but allow some variance
        assert elapsed.endswith("m")
        assert "h" not in elapsed

    def test_elapsed_returns_hours_and_minutes_for_long_duration(self):
        """Test elapsed returns hours and minutes for durations over an hour."""
        agent = Agent(
            name="test",
            session="sess-123",
            worktree=Path("/tmp/test"),
            branch="main",
            started_at=datetime.now() - timedelta(hours=2, minutes=15),
        )
        elapsed = agent.elapsed
        assert "h" in elapsed
        assert "m" in elapsed

    def test_elapsed_zero_minutes(self):
        """Test elapsed when just started."""
        agent = Agent(
            name="test",
            session="sess-123",
            worktree=Path("/tmp/test"),
            branch="main",
            started_at=datetime.now(),
        )
        assert agent.elapsed == "0m"
