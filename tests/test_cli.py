"""Tests for crew.cli module - dashboard command, queue command, and command parsing."""

from __future__ import annotations

import io
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from crew.agent import Agent
from crew.state import State
from crew.cli import (
    cmd_dashboard,
    cmd_queue,
    cmd_ready,
    handle_command,
    get_prompt,
    _next_worker_name,
    recover_session,
    render_dashboard,
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

    def test_handle_command_q_runs_queue(
        self, empty_state: State, project_root: Path
    ):
        """Test handle_command runs queue for 'q' (now alias for queue, not quit)."""
        with patch("crew.cli.cmd_queue") as mock_queue:
            result = handle_command("q", empty_state, project_root)
            assert result is True
            mock_queue.assert_called_once()

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


class TestQueueCommand:
    """Test the queue command for displaying dependency pipeline."""

    def test_queue_no_tickets(self, empty_state: State, capsys):
        """Test queue shows message when no tickets exist."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="[]",
                stderr="",
            )
            cmd_queue(empty_state, [])
            captured = capsys.readouterr()
            assert "No tickets" in captured.out

    def test_queue_no_open_tickets(self, empty_state: State, capsys):
        """Test queue shows message when all tickets are closed."""
        tickets = [
            {"id": "c-1", "status": "closed", "deps": [], "title": "Task 1"},
            {"id": "c-2", "status": "closed", "deps": [], "title": "Task 2"},
        ]
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps(tickets),
                stderr="",
            )
            cmd_queue(empty_state, [])
            captured = capsys.readouterr()
            assert "No open tickets" in captured.out

    def test_queue_ready_tickets(self, empty_state: State, capsys):
        """Test queue displays ready tickets (no dependencies)."""
        tickets = [
            {"id": "c-1", "status": "open", "deps": [], "title": "Ready Task 1"},
            {"id": "c-2", "status": "open", "deps": [], "title": "Ready Task 2"},
        ]
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps(tickets),
                stderr="",
            )
            cmd_queue(empty_state, [])
            captured = capsys.readouterr()
            assert "Ready now" in captured.out
            assert "c-1" in captured.out
            assert "c-2" in captured.out
            assert "Ready Task 1" in captured.out

    def test_queue_next_tickets(self, empty_state: State, capsys):
        """Test queue displays next tickets (blocked by ready tickets)."""
        tickets = [
            {"id": "c-1", "status": "open", "deps": [], "title": "Ready Task"},
            {"id": "c-2", "status": "open", "deps": ["c-1"], "title": "Next Task"},
        ]
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps(tickets),
                stderr="",
            )
            cmd_queue(empty_state, [])
            captured = capsys.readouterr()
            assert "Ready now" in captured.out
            assert "Next" in captured.out
            assert "c-1" in captured.out
            assert "c-2" in captured.out
            assert "blocked by c-1" in captured.out

    def test_queue_later_tickets(self, empty_state: State, capsys):
        """Test queue displays later tickets (multiple deps away)."""
        tickets = [
            {"id": "c-1", "status": "open", "deps": [], "title": "Ready Task"},
            {"id": "c-2", "status": "open", "deps": ["c-1"], "title": "Next Task"},
            {"id": "c-3", "status": "open", "deps": ["c-2"], "title": "Later Task"},
        ]
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps(tickets),
                stderr="",
            )
            cmd_queue(empty_state, [])
            captured = capsys.readouterr()
            assert "Ready now" in captured.out
            assert "Next" in captured.out
            assert "Later" in captured.out
            assert "c-3" in captured.out
            assert "wave 2" in captured.out

    def test_queue_blocked_tickets_cycle(self, empty_state: State, capsys):
        """Test queue displays blocked tickets with cyclic dependencies."""
        tickets = [
            {"id": "c-1", "status": "open", "deps": ["c-2"], "title": "Cycle A"},
            {"id": "c-2", "status": "open", "deps": ["c-1"], "title": "Cycle B"},
        ]
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps(tickets),
                stderr="",
            )
            cmd_queue(empty_state, [])
            captured = capsys.readouterr()
            assert "Blocked" in captured.out
            assert "c-1" in captured.out
            assert "c-2" in captured.out

    def test_queue_ignores_closed_deps(self, empty_state: State, capsys):
        """Test queue ignores closed tickets as dependencies."""
        tickets = [
            {"id": "c-1", "status": "closed", "deps": [], "title": "Closed Task"},
            {"id": "c-2", "status": "open", "deps": ["c-1"], "title": "Open Task"},
        ]
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps(tickets),
                stderr="",
            )
            cmd_queue(empty_state, [])
            captured = capsys.readouterr()
            # c-2 should be ready since c-1 is closed
            assert "Ready now" in captured.out
            assert "c-2" in captured.out

    def test_queue_shows_summary(self, empty_state: State, capsys):
        """Test queue shows summary with counts."""
        tickets = [
            {"id": "c-1", "status": "open", "deps": [], "title": "Task 1"},
            {"id": "c-2", "status": "open", "deps": ["c-1"], "title": "Task 2"},
        ]
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps(tickets),
                stderr="",
            )
            cmd_queue(empty_state, [])
            captured = capsys.readouterr()
            assert "2 open tickets" in captured.out
            assert "1 ready" in captured.out

    def test_queue_tk_not_found(self, empty_state: State, capsys):
        """Test queue handles tk command not found."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()
            cmd_queue(empty_state, [])
            captured = capsys.readouterr()
            assert "tk command not found" in captured.out

    def test_queue_tk_error(self, empty_state: State, capsys):
        """Test queue handles tk query error."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout="",
                stderr="Some error",
            )
            cmd_queue(empty_state, [])
            captured = capsys.readouterr()
            assert "tk query failed" in captured.out

    def test_queue_invalid_json(self, empty_state: State, capsys):
        """Test queue handles invalid JSON from tk."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="not valid json",
                stderr="",
            )
            cmd_queue(empty_state, [])
            captured = capsys.readouterr()
            assert "Failed to parse" in captured.out


class TestQueueCommandRegistration:
    """Test queue command registration in handle_command."""

    def test_handle_command_queue(self, empty_state: State, project_root: Path):
        """Test handle_command accepts 'queue' command."""
        with patch("crew.cli.cmd_queue") as mock_queue:
            result = handle_command("queue", empty_state, project_root)
            assert result is True
            mock_queue.assert_called_once()

    def test_handle_command_q_alias(self, empty_state: State, project_root: Path):
        """Test handle_command accepts 'q' as queue alias."""
        with patch("crew.cli.cmd_queue") as mock_queue:
            result = handle_command("q", empty_state, project_root)
            assert result is True
            mock_queue.assert_called_once()

    def test_quit_still_works(self, empty_state: State, project_root: Path):
        """Test 'quit' still works for exiting."""
        result = handle_command("quit", empty_state, project_root)
        assert result is False

    def test_exit_still_works(self, empty_state: State, project_root: Path):
        """Test 'exit' still works for exiting."""
        result = handle_command("exit", empty_state, project_root)
        assert result is False


class TestReadyCommand:
    """Test the ready command output with agent assignment indicators."""

    def test_ready_no_tickets(self, empty_state: State, capsys):
        """Test ready shows message when no tickets."""
        with patch("crew.cli.run_tk") as mock_run_tk:
            mock_run_tk.return_value = ""
            cmd_ready(empty_state, [])
            captured = capsys.readouterr()
            assert "No ready work" in captured.out

    def test_ready_shows_available_tickets(self, empty_state: State, capsys):
        """Test ready shows available tickets without assignments."""
        with patch("crew.cli.run_tk") as mock_run_tk:
            mock_run_tk.return_value = "c-1234 Add new feature\nc-5678 Fix bug"
            cmd_ready(empty_state, [])
            captured = capsys.readouterr()
            assert "Available" in captured.out
            assert "c-1234" in captured.out
            assert "c-5678" in captured.out

    def test_ready_shows_in_progress_for_assigned_tickets(self, empty_state: State, capsys):
        """Test ready shows In Progress section for assigned tickets."""
        # Add an agent with an assigned task
        agent = Agent(
            name="a",
            session="test-session",
            worktree=Path("/tmp/worktree"),
            branch="agent/a-c-1234",
            task="c-1234",
            status="working",
        )
        empty_state.agents["a"] = agent

        with patch("crew.cli.run_tk") as mock_run_tk:
            mock_run_tk.return_value = "c-1234 Add new feature\nc-5678 Fix bug"
            cmd_ready(empty_state, [])
            captured = capsys.readouterr()
            assert "In Progress" in captured.out
            assert "[a]" in captured.out  # Agent name marker
            assert "Available" in captured.out
            assert "c-5678" in captured.out

    def test_ready_shows_agent_name_in_brackets(self, empty_state: State, capsys):
        """Test assigned tickets show agent name in brackets like [a]."""
        agent = Agent(
            name="b",
            session="test-session",
            worktree=Path("/tmp/worktree"),
            branch="agent/b-c-9999",
            task="c-9999",
            status="ready",
        )
        empty_state.agents["b"] = agent

        with patch("crew.cli.run_tk") as mock_run_tk:
            mock_run_tk.return_value = "c-9999 Some task"
            cmd_ready(empty_state, [])
            captured = capsys.readouterr()
            assert "c-9999 [b]" in captured.out

    def test_ready_all_assigned(self, empty_state: State, capsys):
        """Test when all ready tickets are assigned to agents."""
        agent = Agent(
            name="x",
            session="test-session",
            worktree=Path("/tmp/worktree"),
            branch="agent/x-c-1111",
            task="c-1111",
            status="working",
        )
        empty_state.agents["x"] = agent

        with patch("crew.cli.run_tk") as mock_run_tk:
            mock_run_tk.return_value = "c-1111 Only task"
            cmd_ready(empty_state, [])
            captured = capsys.readouterr()
            assert "In Progress" in captured.out
            assert "Available" not in captured.out

    def test_ready_multiple_agents(self, empty_state: State, capsys):
        """Test ready with multiple agents assigned to different tickets."""
        agent_a = Agent(
            name="a",
            session="test-session-a",
            worktree=Path("/tmp/worktree-a"),
            branch="agent/a-c-1",
            task="c-1",
            status="working",
        )
        agent_b = Agent(
            name="b",
            session="test-session-b",
            worktree=Path("/tmp/worktree-b"),
            branch="agent/b-c-2",
            task="c-2",
            status="working",
        )
        empty_state.agents["a"] = agent_a
        empty_state.agents["b"] = agent_b

        with patch("crew.cli.run_tk") as mock_run_tk:
            mock_run_tk.return_value = "c-1 Task 1\nc-2 Task 2\nc-3 Task 3"
            cmd_ready(empty_state, [])
            captured = capsys.readouterr()
            assert "In Progress" in captured.out
            assert "[a]" in captured.out
            assert "[b]" in captured.out
            assert "Available" in captured.out
            assert "c-3" in captured.out


class TestRecoverSession:
    """Test the recover_session function for session recovery on startup."""

    def test_recover_session_empty_state(self, project_root: Path, empty_state: State):
        """Test recover_session returns False for empty state."""
        result = recover_session(empty_state, project_root)
        assert result is False

    def test_recover_session_with_agents_returns_true(self, project_root: Path, capsys):
        """Test recover_session returns True when agents exist."""
        state = State()
        agent = Agent(
            name="test-agent",
            session="sess-test",
            worktree=None,
            branch="",
            task=None,
            status="idle",
        )
        state.add_agent(agent)

        with patch("crew.cli.get_worktree_list") as mock_worktrees:
            mock_worktrees.return_value = []
            result = recover_session(state, project_root)

        assert result is True

    def test_recover_session_displays_agents(self, project_root: Path, capsys):
        """Test recover_session displays agent status."""
        state = State()
        agent = Agent(
            name="display-agent",
            session="sess-display",
            worktree=None,
            branch="",
            task=None,
            status="idle",
        )
        state.add_agent(agent)

        with patch("crew.cli.get_worktree_list") as mock_worktrees:
            mock_worktrees.return_value = []
            recover_session(state, project_root)

        captured = capsys.readouterr()
        assert "display-agent" in captured.out
        assert "idle" in captured.out

    def test_recover_session_resets_working_agent_with_missing_worktree(
        self, project_root: Path, capsys
    ):
        """Test recover_session resets working agents with missing worktree to idle."""
        state = State()
        agent = Agent(
            name="missing-wt-agent",
            session="sess-missing",
            worktree=Path("/nonexistent/path"),
            branch="agent/missing-wt-agent",
            task="some-task",
            status="working",
            step_count=5,
        )
        state.add_agent(agent)

        with patch("crew.cli.get_worktree_list") as mock_worktrees:
            mock_worktrees.return_value = []
            recover_session(state, project_root)

        # Agent should be reset to idle
        assert agent.status == "idle"
        assert agent.worktree is None
        assert agent.task is None
        assert agent.branch == ""
        assert agent.step_count == 0

        captured = capsys.readouterr()
        assert "Reset missing-wt-agent to idle" in captured.out
        assert "worktree missing" in captured.out

    def test_recover_session_reconciles_working_agent_with_worktree(
        self, project_root: Path, temp_dir: Path, capsys
    ):
        """Test recover_session reconciles working agents with existing worktree."""
        # Create a fake worktree directory
        worktree_path = temp_dir / "agents" / "test-wt"
        worktree_path.mkdir(parents=True)

        state = State()
        agent = Agent(
            name="working-agent",
            session="sess-working",
            worktree=worktree_path,
            branch="agent/working-agent",
            task="some-task",
            status="working",
            step_count=3,
        )
        state.add_agent(agent)

        with patch("crew.cli.get_worktree_list") as mock_worktrees:
            mock_worktrees.return_value = [{"path": str(worktree_path)}]
            with patch("crew.cli.has_uncommitted_changes") as mock_dirty:
                mock_dirty.return_value = False  # Clean worktree
                with patch("crew.cli.shutdown_agent") as mock_shutdown:
                    mock_shutdown.return_value = "partial"
                    with patch("crew.cli.remove_worktree"):
                        recover_session(state, temp_dir)

        # shutdown_agent should have been called
        mock_shutdown.assert_called_once()
        captured = capsys.readouterr()
        assert "partial work" in captured.out

    def test_recover_session_warns_orphaned_worktrees(self, temp_dir: Path, capsys):
        """Test recover_session warns about orphaned worktrees."""
        # Create orphaned worktree (exists on disk but not in state)
        orphan_dir = temp_dir / "agents" / "orphan"
        orphan_dir.mkdir(parents=True)

        # Create .crew dir
        (temp_dir / ".crew").mkdir(parents=True)

        state = State()
        # Add an idle agent (no worktree)
        agent = Agent(
            name="idle-agent",
            session="",
            worktree=None,
            branch="",
            task=None,
            status="idle",
        )
        state.add_agent(agent)

        with patch("crew.cli.get_worktree_list") as mock_worktrees:
            mock_worktrees.return_value = []
            recover_session(state, temp_dir)

        captured = capsys.readouterr()
        assert "Orphaned worktrees" in captured.out
        # Check the path is in output (may be split by console wrapping)
        assert "agents" in captured.out and "orphan" in captured.out

    def test_recover_session_resets_done_agent_with_missing_worktree(self, project_root: Path, capsys):
        """Test recover_session resets done agents with missing worktree to idle."""
        state = State()
        agent = Agent(
            name="done-agent",
            session="sess-done",
            worktree=None,
            branch="agent/done-agent",
            task="completed-task",
            status="done",
        )
        state.add_agent(agent)

        with patch("crew.cli.get_worktree_list") as mock_worktrees:
            mock_worktrees.return_value = []
            recover_session(state, project_root)

        captured = capsys.readouterr()
        assert "done-agent" in captured.out
        assert "done but worktree missing" in captured.out
        # Agent should be reset to idle since worktree is gone
        assert agent.status == "idle"
        assert agent.worktree is None
        assert agent.task is None

    def test_recover_session_completes_done_agent_with_worktree(self, temp_dir: Path, capsys):
        """Test recover_session calls complete_task for done agents with existing worktree."""
        worktree_path = temp_dir / "agents" / "done-with-wt"
        worktree_path.mkdir(parents=True)
        (temp_dir / ".crew").mkdir(parents=True)

        state = State()
        agent = Agent(
            name="done-wt-agent",
            session="sess-done-wt",
            worktree=worktree_path,
            branch="agent/done-wt-agent",
            task="task-to-complete",
            status="done",
        )
        state.add_agent(agent)

        with patch("crew.cli.get_worktree_list") as mock_worktrees:
            mock_worktrees.return_value = [{"path": str(worktree_path)}]
            with patch("crew.cli.complete_task") as mock_complete:
                recover_session(state, temp_dir)
                # complete_task should be called for the done agent with worktree
                mock_complete.assert_called_once()

        captured = capsys.readouterr()
        assert "done-wt-agent" in captured.out
        assert "done, pending merge" in captured.out
        assert "Completed done-wt-agent" in captured.out

    def test_recover_session_shows_stuck_agents(self, project_root: Path, capsys):
        """Test recover_session displays stuck agents correctly."""
        state = State()
        agent = Agent(
            name="stuck-agent",
            session="sess-stuck",
            worktree=None,
            branch="agent/stuck-agent",
            task="stuck-task",
            status="stuck",
        )
        state.add_agent(agent)

        with patch("crew.cli.get_worktree_list") as mock_worktrees:
            mock_worktrees.return_value = []
            recover_session(state, project_root)

        captured = capsys.readouterr()
        assert "stuck-agent" in captured.out
        assert "stuck" in captured.out

    def test_recover_session_saves_state(self, project_root: Path):
        """Test recover_session saves state after reconciliation."""
        state = State()
        agent = Agent(
            name="save-test",
            session="",
            worktree=None,
            branch="",
            task=None,
            status="idle",
        )
        state.add_agent(agent)

        with patch("crew.cli.get_worktree_list") as mock_worktrees:
            mock_worktrees.return_value = []
            with patch("crew.cli.save_state") as mock_save:
                recover_session(state, project_root)
                mock_save.assert_called_once_with(state, project_root)

    def test_recover_session_shows_summary_working(self, project_root: Path, temp_dir: Path, capsys):
        """Test recover_session shows correct summary for working agents."""
        worktree_path = temp_dir / "agents" / "test-wt"
        worktree_path.mkdir(parents=True)

        state = State()
        agent = Agent(
            name="working-sum",
            session="sess-sum",
            worktree=worktree_path,
            branch="agent/working-sum",
            task="task-sum",
            status="working",
        )
        state.add_agent(agent)

        with patch("crew.cli.get_worktree_list") as mock_worktrees:
            mock_worktrees.return_value = [{"path": str(worktree_path)}]
            with patch("crew.cli.has_uncommitted_changes") as mock_dirty:
                mock_dirty.return_value = False  # Clean worktree
                with patch("crew.cli.shutdown_agent") as mock_shutdown:
                    # Return "done" so agent stays in a working state for the summary
                    mock_shutdown.return_value = "done"
                    recover_session(state, temp_dir)

        captured = capsys.readouterr()
        # Agent marked as done should be visible (1 done agent)
        assert "done" in captured.out.lower()

    def test_recover_session_shows_summary_idle(self, project_root: Path, capsys):
        """Test recover_session shows correct summary for idle agents."""
        state = State()
        agent = Agent(
            name="idle-sum",
            session="",
            worktree=None,
            branch="",
            task=None,
            status="idle",
        )
        state.add_agent(agent)

        with patch("crew.cli.get_worktree_list") as mock_worktrees:
            mock_worktrees.return_value = []
            recover_session(state, project_root)

        captured = capsys.readouterr()
        assert "Session restored" in captured.out or "idle" in captured.out

    def test_recover_session_handles_worktree_list_error(self, project_root: Path, capsys):
        """Test recover_session handles errors when listing worktrees."""
        state = State()
        agent = Agent(
            name="error-test",
            session="",
            worktree=None,
            branch="",
            task=None,
            status="idle",
        )
        state.add_agent(agent)

        with patch("crew.cli.get_worktree_list") as mock_worktrees:
            mock_worktrees.side_effect = Exception("Git error")
            # Should not crash
            result = recover_session(state, project_root)

        assert result is True
        captured = capsys.readouterr()
        assert "Could not list worktrees" in captured.out

    def test_recover_session_completes_done_from_logs(self, temp_dir: Path, capsys):
        """Test recover_session calls complete_task when DONE found in logs."""
        worktree_path = temp_dir / "agents" / "done-logs"
        worktree_path.mkdir(parents=True)
        (temp_dir / ".crew").mkdir(parents=True)

        state = State()
        agent = Agent(
            name="done-logs-agent",
            session="sess-done-logs",
            worktree=worktree_path,
            branch="agent/done-logs-agent",
            task="task-done",
            status="working",
        )
        state.add_agent(agent)

        with patch("crew.cli.get_worktree_list") as mock_worktrees:
            mock_worktrees.return_value = [{"path": str(worktree_path)}]
            with patch("crew.cli.has_uncommitted_changes") as mock_dirty:
                mock_dirty.return_value = False  # Clean worktree
                with patch("crew.cli.shutdown_agent") as mock_shutdown:
                    mock_shutdown.return_value = "done"
                    with patch("crew.cli.complete_task") as mock_complete:
                        recover_session(state, temp_dir)
                        # complete_task should be called for the done agent
                        mock_complete.assert_called_once()

        captured = capsys.readouterr()
        assert "Completed done-logs-agent" in captured.out

    def test_recover_session_resets_dirty_worktree_to_idle(
        self, temp_dir: Path, capsys
    ):
        """Test recover_session resets agents with dirty worktree to idle."""
        worktree_path = temp_dir / "agents" / "dirty-wt"
        worktree_path.mkdir(parents=True)
        (temp_dir / ".crew").mkdir(parents=True)

        state = State()
        agent = Agent(
            name="dirty-agent",
            session="sess-dirty",
            worktree=worktree_path,
            branch="agent/dirty-agent",
            task="task-dirty",
            status="working",
            step_count=5,
        )
        state.add_agent(agent)

        with patch("crew.cli.get_worktree_list") as mock_worktrees:
            mock_worktrees.return_value = [{"path": str(worktree_path)}]
            with patch("crew.cli.has_uncommitted_changes") as mock_dirty:
                mock_dirty.return_value = True  # Dirty worktree
                with patch("crew.cli.remove_worktree") as mock_remove:
                    recover_session(state, temp_dir)

        # Worktree should have been removed
        mock_remove.assert_called_once()
        # Agent should be reset to idle
        assert agent.status == "idle"
        assert agent.worktree is None
        assert agent.task is None
        assert agent.branch == ""
        assert agent.step_count == 0
        # Output should mention dirty worktree
        captured = capsys.readouterr()
        assert "dirty worktree removed" in captured.out
        # Check parts separately as console may wrap text
        assert "ticket task-dirty stays" in captured.out

    def test_recover_session_resets_ready_with_missing_worktree(
        self, project_root: Path, capsys
    ):
        """Test recover_session resets ready agents with missing worktree to idle."""
        state = State()
        agent = Agent(
            name="ready-missing",
            session="sess-ready",
            worktree=Path("/nonexistent/ready"),
            branch="agent/ready-missing",
            task="ready-task",
            status="ready",
            step_count=0,
        )
        state.add_agent(agent)

        with patch("crew.cli.get_worktree_list") as mock_worktrees:
            mock_worktrees.return_value = []
            recover_session(state, project_root)

        # Agent should be reset to idle
        assert agent.status == "idle"
        assert agent.worktree is None

        captured = capsys.readouterr()
        assert "Reset ready-missing to idle" in captured.out


class TestRenderDashboard:
    """Test the render_dashboard function."""

    def test_render_dashboard_returns_group(self, project_root: Path, empty_state: State):
        """Test render_dashboard returns a Rich Group."""
        from rich.console import Group

        result = render_dashboard(empty_state, project_root, runner_active=False)
        assert isinstance(result, Group)

    def test_render_dashboard_runner_stopped(self, project_root: Path, empty_state: State):
        """Test render_dashboard shows runner stopped when not active."""
        from rich.console import Console
        from io import StringIO

        console = Console(file=StringIO(), force_terminal=True)
        result = render_dashboard(empty_state, project_root, runner_active=False)
        console.print(result)
        output = console.file.getvalue()
        assert "Runner stopped" in output

    def test_render_dashboard_runner_active(self, project_root: Path, empty_state: State):
        """Test render_dashboard shows runner active when running."""
        from rich.console import Console
        from io import StringIO

        console = Console(file=StringIO(), force_terminal=True)
        result = render_dashboard(empty_state, project_root, runner_active=True)
        console.print(result)
        output = console.file.getvalue()
        assert "Runner active" in output

    def test_render_dashboard_no_agents(self, project_root: Path, empty_state: State):
        """Test render_dashboard shows 'No agents' when empty."""
        from rich.console import Console
        from io import StringIO

        console = Console(file=StringIO(), force_terminal=True)
        result = render_dashboard(empty_state, project_root, runner_active=False)
        console.print(result)
        output = console.file.getvalue()
        assert "No agents" in output

    def test_render_dashboard_with_agents(self, project_root: Path):
        """Test render_dashboard includes agent information."""
        from rich.console import Console
        from io import StringIO

        state = State()
        agent = Agent(
            name="render-test",
            session="sess",
            worktree=Path("/tmp/test"),
            branch="main",
            task="Test task",
            status="working",
            total_input_tokens=1000,
            total_output_tokens=500,
            total_cost_usd=0.05,
        )
        state.add_agent(agent)

        with patch("crew.cli.read_log_tail") as mock_log:
            mock_log.return_value = None
            console = Console(file=StringIO(), force_terminal=True)
            result = render_dashboard(state, project_root, runner_active=False)
            console.print(result)
            output = console.file.getvalue()

        assert "render-test" in output
        assert "Test task" in output
        assert "working" in output


class TestDashboardLiveMode:
    """Test the dashboard live mode flag parsing."""

    def test_dashboard_accepts_l_flag(self, empty_state: State, project_root: Path):
        """Test dashboard accepts -l flag for live mode."""
        with patch("crew.cli._run_live_dashboard") as mock_live:
            cmd_dashboard(empty_state, ["-l"], project_root)
            mock_live.assert_called_once_with(empty_state, project_root)

    def test_dashboard_accepts_live_flag(self, empty_state: State, project_root: Path):
        """Test dashboard accepts --live flag for live mode."""
        with patch("crew.cli._run_live_dashboard") as mock_live:
            cmd_dashboard(empty_state, ["--live"], project_root)
            mock_live.assert_called_once_with(empty_state, project_root)

    def test_dashboard_without_flag_not_live(self, empty_state: State, project_root: Path):
        """Test dashboard without flag does not run in live mode."""
        with patch("crew.cli._run_live_dashboard") as mock_live:
            cmd_dashboard(empty_state, [], project_root)
            mock_live.assert_not_called()

    def test_handle_command_dashboard_l_flag(self, empty_state: State, project_root: Path):
        """Test handle_command passes -l flag to dashboard."""
        with patch("crew.cli._run_live_dashboard") as mock_live:
            handle_command("dashboard -l", empty_state, project_root)
            mock_live.assert_called_once()

    def test_handle_command_d_l_flag(self, empty_state: State, project_root: Path):
        """Test handle_command passes -l flag with d shortcut."""
        with patch("crew.cli._run_live_dashboard") as mock_live:
            handle_command("d -l", empty_state, project_root)
            mock_live.assert_called_once()

    def test_handle_command_s_live_flag(self, empty_state: State, project_root: Path):
        """Test handle_command passes --live flag with s shortcut."""
        with patch("crew.cli._run_live_dashboard") as mock_live:
            handle_command("s --live", empty_state, project_root)
            mock_live.assert_called_once()
