"""Tests for crew.cli module - dashboard command and command parsing."""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from crew.agent import Agent
from crew.state import State
from crew.cli import (
    cmd_dashboard,
    handle_command,
    get_prompt,
    _next_worker_name,
)


@pytest.fixture
def mock_git_status_panels():
    """Mock print_git_status_panels to avoid git calls on fake worktrees."""
    with patch("crew.cli.print_git_status_panels"):
        yield


@pytest.fixture
def mock_read_log_tail():
    """Mock read_log_tail to avoid file system access."""
    with patch("crew.cli.read_log_tail") as mock:
        mock.return_value = None
        yield mock


class TestDashboardCommand:
    """Test the dashboard command output."""

    def test_dashboard_with_no_agents(self, project_root: Path, empty_state: State, capsys):
        """Test dashboard shows 'No agents' when state is empty."""
        cmd_dashboard(empty_state, [], project_root)
        captured = capsys.readouterr()
        assert "No agents" in captured.out

    def test_dashboard_shows_runner_stopped_when_not_running(
        self, project_root: Path, empty_state: State, capsys
    ):
        """Test dashboard shows runner stopped when runner is not active."""
        cmd_dashboard(empty_state, [], project_root)
        captured = capsys.readouterr()
        assert "Runner stopped" in captured.out

    @pytest.mark.usefixtures("mock_git_status_panels", "mock_read_log_tail")
    def test_dashboard_shows_session_totals(
        self, project_root: Path, populated_state: State, capsys
    ):
        """Test dashboard shows session token and cost totals."""
        cmd_dashboard(populated_state, [], project_root)
        captured = capsys.readouterr()
        assert "Session:" in captured.out
        assert "tokens" in captured.out
        assert "$" in captured.out

    @pytest.mark.usefixtures("mock_git_status_panels", "mock_read_log_tail")
    def test_dashboard_shows_agent_table_columns(
        self, project_root: Path, populated_state: State, capsys
    ):
        """Test dashboard shows expected table columns."""
        cmd_dashboard(populated_state, [], project_root)
        captured = capsys.readouterr()
        assert "NAME" in captured.out
        assert "TASK" in captured.out
        assert "STATUS" in captured.out
        assert "STEPS" in captured.out
        assert "TOKENS" in captured.out
        assert "COST" in captured.out

    @pytest.mark.usefixtures("mock_git_status_panels", "mock_read_log_tail")
    def test_dashboard_shows_agent_names(
        self, project_root: Path, populated_state: State, capsys
    ):
        """Test dashboard displays agent names."""
        cmd_dashboard(populated_state, [], project_root)
        captured = capsys.readouterr()
        for agent in populated_state.agents.values():
            assert agent.name in captured.out

    @pytest.mark.usefixtures("mock_git_status_panels", "mock_read_log_tail")
    def test_dashboard_shows_agent_status(
        self, project_root: Path, populated_state: State, capsys
    ):
        """Test dashboard displays agent statuses."""
        cmd_dashboard(populated_state, [], project_root)
        captured = capsys.readouterr()
        for agent in populated_state.agents.values():
            assert agent.status in captured.out

    @pytest.mark.usefixtures("mock_git_status_panels", "mock_read_log_tail")
    def test_dashboard_shows_agent_task(
        self, project_root: Path, capsys
    ):
        """Test dashboard displays agent task when assigned."""
        state = State()
        agent = Agent(
            name="task-agent",
            session="sess-task",
            worktree=Path("/tmp/task"),
            branch="main",
            task="Fix the bug",
            status="working",
        )
        state.add_agent(agent)

        cmd_dashboard(state, [], project_root)
        captured = capsys.readouterr()
        assert "Fix the bug" in captured.out

    def test_dashboard_shows_dash_for_no_task(
        self, project_root: Path, capsys
    ):
        """Test dashboard shows '-' when no task is assigned."""
        state = State()
        agent = Agent(
            name="idle-agent",
            session="sess-idle",
            worktree=Path("/tmp/idle"),
            branch="main",
            task=None,
            status="idle",
        )
        state.add_agent(agent)

        cmd_dashboard(state, [], project_root)
        captured = capsys.readouterr()
        # The table should contain a dash for missing task
        assert "-" in captured.out

    @pytest.mark.usefixtures("mock_git_status_panels", "mock_read_log_tail")
    def test_dashboard_shows_step_count(
        self, project_root: Path, capsys
    ):
        """Test dashboard displays step count."""
        state = State()
        agent = Agent(
            name="stepped-agent",
            session="sess-stepped",
            worktree=Path("/tmp/stepped"),
            branch="main",
            status="working",
            step_count=42,
        )
        state.add_agent(agent)

        cmd_dashboard(state, [], project_root)
        captured = capsys.readouterr()
        assert "42" in captured.out

    @pytest.mark.usefixtures("mock_git_status_panels", "mock_read_log_tail")
    def test_dashboard_shows_token_count(
        self, project_root: Path, capsys
    ):
        """Test dashboard displays token counts."""
        state = State()
        agent = Agent(
            name="token-agent",
            session="sess-token",
            worktree=Path("/tmp/tokens"),
            branch="main",
            status="working",
            total_input_tokens=5000,
            total_output_tokens=2500,
        )
        state.add_agent(agent)

        cmd_dashboard(state, [], project_root)
        captured = capsys.readouterr()
        # Total tokens should be 7500
        assert "7,500" in captured.out

    @pytest.mark.usefixtures("mock_git_status_panels", "mock_read_log_tail")
    def test_dashboard_shows_cost(
        self, project_root: Path, capsys
    ):
        """Test dashboard displays cost information."""
        state = State()
        agent = Agent(
            name="cost-agent",
            session="sess-cost",
            worktree=Path("/tmp/cost"),
            branch="main",
            status="working",
            total_cost_usd=0.1234,
        )
        state.add_agent(agent)

        cmd_dashboard(state, [], project_root)
        captured = capsys.readouterr()
        assert "$0.1234" in captured.out

    def test_dashboard_calculates_total_cost(
        self, project_root: Path, capsys
    ):
        """Test dashboard calculates total cost across agents."""
        state = State()
        agent1 = Agent(
            name="agent1",
            session="sess-1",
            worktree=Path("/tmp/1"),
            branch="main",
            total_cost_usd=0.50,
        )
        agent2 = Agent(
            name="agent2",
            session="sess-2",
            worktree=Path("/tmp/2"),
            branch="main",
            total_cost_usd=0.25,
        )
        state.add_agent(agent1)
        state.add_agent(agent2)

        cmd_dashboard(state, [], project_root)
        captured = capsys.readouterr()
        # Total should be $0.75
        assert "$0.75" in captured.out

    @pytest.mark.usefixtures("mock_git_status_panels", "mock_read_log_tail")
    def test_dashboard_shows_runner_active_when_running(
        self, project_root: Path, capsys
    ):
        """Test dashboard shows runner active when runner is running."""
        with patch("crew.cli._runner") as mock_runner:
            mock_runner.is_running = True
            mock_runner.drain_events.return_value = []

            state = State()
            agent = Agent(
                name="active-agent",
                session="sess-active",
                worktree=Path("/tmp/active"),
                branch="main",
                status="working",
            )
            state.add_agent(agent)

            cmd_dashboard(state, [], project_root)
            captured = capsys.readouterr()
            assert "Runner active" in captured.out


class TestDashboardWorkingAgentPanels:
    """Test dashboard log tail panels for working agents."""

    @pytest.mark.usefixtures("mock_git_status_panels")
    def test_dashboard_shows_log_panels_for_working_agents(
        self, project_root: Path, capsys
    ):
        """Test dashboard shows log panels for working agents."""
        with patch("crew.cli.read_log_tail") as mock_read_log:
            mock_read_log.return_value = "Some log content here"

            state = State()
            agent = Agent(
                name="working-panel",
                session="sess-panel",
                worktree=Path("/tmp/panel"),
                branch="main",
                task="Test task",
                status="working",
            )
            state.add_agent(agent)

            cmd_dashboard(state, [], project_root)
            captured = capsys.readouterr()
            # Should show agent name in panel
            assert "working-panel" in captured.out

    @pytest.mark.usefixtures("mock_git_status_panels")
    def test_dashboard_truncates_long_log_lines(
        self, project_root: Path, capsys
    ):
        """Test dashboard truncates long log lines."""
        with patch("crew.cli.read_log_tail") as mock_read_log:
            long_line = "x" * 150
            mock_read_log.return_value = long_line

            state = State()
            agent = Agent(
                name="long-log",
                session="sess-long",
                worktree=Path("/tmp/long"),
                branch="main",
                status="working",
            )
            state.add_agent(agent)

            cmd_dashboard(state, [], project_root)
            captured = capsys.readouterr()
            # Line should be truncated (100 chars + "...")
            assert "..." in captured.out


class TestCommandParsing:
    """Test command parsing and handle_command function."""

    def test_handle_command_returns_true_for_empty_input(
        self, empty_state: State, project_root: Path
    ):
        """Test handle_command returns True for empty input."""
        result = handle_command("", empty_state, project_root)
        assert result is True

    def test_handle_command_returns_true_for_whitespace(
        self, empty_state: State, project_root: Path
    ):
        """Test handle_command returns True for whitespace only."""
        result = handle_command("   ", empty_state, project_root)
        assert result is True

    def test_handle_command_quit_returns_false(
        self, empty_state: State, project_root: Path
    ):
        """Test handle_command returns False for quit command."""
        result = handle_command("quit", empty_state, project_root)
        assert result is False

    def test_handle_command_q_returns_false(
        self, empty_state: State, project_root: Path
    ):
        """Test handle_command returns False for 'q' shortcut."""
        result = handle_command("q", empty_state, project_root)
        assert result is False

    def test_handle_command_exit_returns_false(
        self, empty_state: State, project_root: Path
    ):
        """Test handle_command returns False for 'exit' command."""
        result = handle_command("exit", empty_state, project_root)
        assert result is False

    def test_handle_command_dashboard_returns_true(
        self, empty_state: State, project_root: Path
    ):
        """Test handle_command returns True for dashboard command."""
        result = handle_command("dashboard", empty_state, project_root)
        assert result is True

    def test_handle_command_d_shortcut(
        self, empty_state: State, project_root: Path
    ):
        """Test handle_command accepts 'd' as dashboard shortcut."""
        result = handle_command("d", empty_state, project_root)
        assert result is True

    def test_handle_command_s_shortcut(
        self, empty_state: State, project_root: Path
    ):
        """Test handle_command accepts 's' as dashboard shortcut."""
        result = handle_command("s", empty_state, project_root)
        assert result is True

    def test_handle_command_help_returns_true(
        self, empty_state: State, project_root: Path
    ):
        """Test handle_command returns True for help command."""
        result = handle_command("help", empty_state, project_root)
        assert result is True

    def test_handle_command_h_shortcut(
        self, empty_state: State, project_root: Path
    ):
        """Test handle_command accepts 'h' as help shortcut."""
        result = handle_command("h", empty_state, project_root)
        assert result is True

    def test_handle_command_case_insensitive(
        self, empty_state: State, project_root: Path
    ):
        """Test handle_command is case insensitive."""
        result = handle_command("QUIT", empty_state, project_root)
        assert result is False

        result = handle_command("Dashboard", empty_state, project_root)
        assert result is True

    def test_handle_command_unknown_command(
        self, empty_state: State, project_root: Path, capsys
    ):
        """Test handle_command shows error for unknown command."""
        result = handle_command("unknowncommand", empty_state, project_root)
        assert result is True
        captured = capsys.readouterr()
        assert "Unknown command" in captured.out

    def test_handle_command_ready_alias(
        self, empty_state: State, project_root: Path
    ):
        """Test handle_command accepts 'r' as ready shortcut."""
        # ready command runs tk, which may not be installed, but we just test it doesn't crash
        result = handle_command("r", empty_state, project_root)
        assert result is True


class TestGetPrompt:
    """Test the get_prompt function for REPL prompt generation."""

    def test_get_prompt_no_workers(self, empty_state: State):
        """Test get_prompt with no workers returns basic prompt."""
        prompt = get_prompt(empty_state)
        assert prompt == "crew> "

    def test_get_prompt_with_workers_no_runner(self, populated_state: State):
        """Test get_prompt shows worker count when runner not active."""
        prompt = get_prompt(populated_state)
        assert "workers" in prompt
        assert "crew>" in prompt

    def test_get_prompt_runner_active_working(self, populated_state: State):
        """Test get_prompt shows working count when runner active."""
        with patch("crew.cli._runner") as mock_runner:
            mock_runner.is_running = True

            # Ensure we have working agents
            for agent in populated_state.agents.values():
                if agent.status == "working":
                    break
            else:
                # Add a working agent if none exists
                working = Agent(
                    name="work",
                    session="sess",
                    worktree=Path("/tmp/w"),
                    branch="main",
                    status="working",
                )
                populated_state.add_agent(working)

            prompt = get_prompt(populated_state)
            assert "working" in prompt
            assert "crew>" in prompt

    def test_get_prompt_runner_active_idle(self):
        """Test get_prompt shows idle count when all agents idle."""
        with patch("crew.cli._runner") as mock_runner:
            mock_runner.is_running = True

            state = State()
            idle_agent = Agent(
                name="idle",
                session="sess",
                worktree=Path("/tmp/idle"),
                branch="main",
                status="idle",
            )
            state.add_agent(idle_agent)

            prompt = get_prompt(state)
            assert "waiting" in prompt or "idle" in prompt
            assert "crew>" in prompt


class TestNextWorkerName:
    """Test the _next_worker_name function for auto-naming workers."""

    def test_next_worker_name_empty_state(self, empty_state: State):
        """Test first worker gets name 'a'."""
        name = _next_worker_name(empty_state)
        assert name == "a"

    def test_next_worker_name_skips_existing(self, empty_state: State):
        """Test next worker name skips existing names."""
        agent_a = Agent(
            name="a",
            session="sess-a",
            worktree=Path("/tmp/a"),
            branch="main",
        )
        empty_state.add_agent(agent_a)

        name = _next_worker_name(empty_state)
        assert name == "b"

    def test_next_worker_name_finds_gap(self, empty_state: State):
        """Test next worker name finds gaps in sequence."""
        # Add agents a and c, leaving b free
        for letter in ["a", "c"]:
            agent = Agent(
                name=letter,
                session=f"sess-{letter}",
                worktree=Path(f"/tmp/{letter}"),
                branch="main",
            )
            empty_state.add_agent(agent)

        name = _next_worker_name(empty_state)
        assert name == "b"

    def test_next_worker_name_all_letters_used(self, empty_state: State):
        """Test fallback when all letters used."""
        # Add all 26 letters
        for i in range(26):
            letter = chr(ord("a") + i)
            agent = Agent(
                name=letter,
                session=f"sess-{letter}",
                worktree=Path(f"/tmp/{letter}"),
                branch="main",
            )
            empty_state.add_agent(agent)

        name = _next_worker_name(empty_state)
        assert name == "a0"


class TestDashboardStatusColors:
    """Test that dashboard displays correct status styling."""

    def test_dashboard_idle_status(self, project_root: Path, capsys):
        """Test idle agents are displayed."""
        state = State()
        agent = Agent(
            name="idle-test",
            session="sess",
            worktree=Path("/tmp/idle"),
            branch="main",
            status="idle",
        )
        state.add_agent(agent)

        cmd_dashboard(state, [], project_root)
        captured = capsys.readouterr()
        assert "idle" in captured.out

    @pytest.mark.usefixtures("mock_git_status_panels", "mock_read_log_tail")
    def test_dashboard_working_status(self, project_root: Path, capsys):
        """Test working agents are displayed."""
        state = State()
        agent = Agent(
            name="working-test",
            session="sess",
            worktree=Path("/tmp/working"),
            branch="main",
            status="working",
        )
        state.add_agent(agent)

        cmd_dashboard(state, [], project_root)
        captured = capsys.readouterr()
        assert "working" in captured.out

    def test_dashboard_done_status(self, project_root: Path, capsys):
        """Test done agents are displayed."""
        state = State()
        agent = Agent(
            name="done-test",
            session="sess",
            worktree=Path("/tmp/done"),
            branch="main",
            status="done",
        )
        state.add_agent(agent)

        cmd_dashboard(state, [], project_root)
        captured = capsys.readouterr()
        assert "done" in captured.out

    def test_dashboard_stuck_status(self, project_root: Path, capsys):
        """Test stuck agents are displayed."""
        state = State()
        agent = Agent(
            name="stuck-test",
            session="sess",
            worktree=Path("/tmp/stuck"),
            branch="main",
            status="stuck",
        )
        state.add_agent(agent)

        cmd_dashboard(state, [], project_root)
        captured = capsys.readouterr()
        assert "stuck" in captured.out

    @pytest.mark.usefixtures("mock_git_status_panels", "mock_read_log_tail")
    def test_dashboard_ready_status(self, project_root: Path, capsys):
        """Test ready agents are displayed."""
        state = State()
        agent = Agent(
            name="ready-test",
            session="sess",
            worktree=Path("/tmp/ready"),
            branch="main",
            status="ready",
        )
        state.add_agent(agent)

        cmd_dashboard(state, [], project_root)
        captured = capsys.readouterr()
        assert "ready" in captured.out
