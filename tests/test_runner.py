"""Unit tests for crew/runner.py with mocked subprocess."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from crew.agent import Agent
from crew.runner import (
    assign_task,
    check_work_completed,
    complete_task,
    find_claude_process,
    is_done,
    kill_claude_process,
    run_claude,
    shutdown_agent,
    spawn_worker,
    step_agent,
)
from crew.state import State, save_state, load_state


class TestRunClaude:
    """Tests for run_claude function."""

    def test_run_claude_parses_json_response(self, temp_dir: Path):
        """run_claude parses JSON output correctly."""
        mock_response = {
            "result": "Task completed successfully",
            "usage": {"input_tokens": 100, "output_tokens": 50},
            "total_cost_usd": 0.005,
        }

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout=json.dumps(mock_response),
                stderr="",
            )

            result = run_claude("Test prompt", cwd=temp_dir)

            assert result["result"] == "Task completed successfully"
            assert result["input_tokens"] == 100
            assert result["output_tokens"] == 50
            assert result["cost_usd"] == 0.005
            assert result["stderr"] == ""

    def test_run_claude_with_session_new(self, temp_dir: Path):
        """run_claude creates new session with --session-id."""
        mock_response = {"result": "OK", "usage": {}, "total_cost_usd": 0.0}

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout=json.dumps(mock_response),
                stderr="",
            )

            run_claude(
                "Test prompt",
                cwd=temp_dir,
                session="test-session-123",
                is_new_session=True,
            )

            call_args = mock_run.call_args
            cmd = call_args[0][0]
            assert "--session-id" in cmd
            assert "test-session-123" in cmd

    def test_run_claude_resumes_existing_session(self, temp_dir: Path):
        """run_claude resumes existing session with --resume."""
        mock_response = {"result": "OK", "usage": {}, "total_cost_usd": 0.0}

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout=json.dumps(mock_response),
                stderr="",
            )

            run_claude(
                "Test prompt",
                cwd=temp_dir,
                session="test-session-123",
                is_new_session=False,
            )

            call_args = mock_run.call_args
            cmd = call_args[0][0]
            assert "--resume" in cmd
            assert "test-session-123" in cmd

    def test_run_claude_handles_invalid_json(self, temp_dir: Path):
        """run_claude falls back to raw output if JSON is invalid."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="Not valid JSON output",
                stderr="",
            )

            result = run_claude("Test prompt", cwd=temp_dir)

            assert result["result"] == "Not valid JSON output"
            assert result["input_tokens"] == 0
            assert result["output_tokens"] == 0

    def test_run_claude_timeout_raises_error(self, temp_dir: Path):
        """run_claude raises TimeoutError on timeout."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=300)

            with pytest.raises(TimeoutError) as exc_info:
                run_claude("Test prompt", cwd=temp_dir, timeout=300)

            assert "timed out" in str(exc_info.value)

    def test_run_claude_not_found_raises_error(self, temp_dir: Path):
        """run_claude raises RuntimeError if Claude CLI not found."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()

            with pytest.raises(RuntimeError) as exc_info:
                run_claude("Test prompt", cwd=temp_dir)

            assert "Claude CLI not found" in str(exc_info.value)


class TestIsDone:
    """Tests for is_done function."""

    def test_is_done_with_done_on_own_line(self):
        """is_done returns True when DONE is on its own line."""
        assert is_done("Some output\nDONE\nMore output") is True

    def test_is_done_case_insensitive(self):
        """is_done is case insensitive."""
        assert is_done("done") is True
        assert is_done("Done") is True
        assert is_done("DONE") is True

    def test_is_done_with_whitespace(self):
        """is_done handles whitespace around DONE."""
        assert is_done("  DONE  ") is True
        assert is_done("\nDONE\n") is True

    def test_is_done_returns_false_for_embedded_done(self):
        """is_done returns False when DONE is part of other text."""
        assert is_done("Not DONE yet") is False
        assert is_done("UNDONE") is False

    def test_is_done_returns_false_for_no_done(self):
        """is_done returns False when no DONE present."""
        assert is_done("Still working on task") is False
        assert is_done("") is False


class TestSpawnWorker:
    """Tests for spawn_worker function."""

    def test_spawn_worker_creates_idle_agent(self, project_root: Path):
        """spawn_worker creates an agent in idle state."""
        state = State()

        agent = spawn_worker("test-agent", state, project_root=project_root)

        assert agent.name == "test-agent"
        assert agent.status == "idle"
        assert agent.session == ""
        assert agent.worktree is None
        assert agent.branch == ""
        assert agent.task is None
        assert agent.step_count == 0

    def test_spawn_worker_adds_agent_to_state(self, project_root: Path):
        """spawn_worker adds the agent to state."""
        state = State()

        agent = spawn_worker("test-agent", state, project_root=project_root)

        assert state.get_agent("test-agent") is agent

    def test_spawn_worker_initializes_token_counters(self, project_root: Path):
        """spawn_worker initializes token counters to zero."""
        state = State()

        agent = spawn_worker("test-agent", state, project_root=project_root)

        assert agent.total_input_tokens == 0
        assert agent.total_output_tokens == 0
        assert agent.total_cost_usd == 0.0


class TestAssignTask:
    """Tests for assign_task function."""

    def test_assign_task_to_idle_agent(self, project_root: Path):
        """assign_task sets up worktree and task for idle agent."""
        state = State()
        agent = spawn_worker("test-agent", state, project_root=project_root)

        # Mock create_worktree and get_task_description
        with patch("crew.runner.create_worktree") as mock_create_wt, \
             patch("crew.runner.get_task_description") as mock_get_desc, \
             patch("crew.runner.generate_session_id") as mock_gen_session:

            worktree_path = project_root / "agents" / "test-agent-c-123"
            worktree_path.mkdir(parents=True, exist_ok=True)
            mock_create_wt.return_value = worktree_path
            mock_get_desc.return_value = "Test task description"
            mock_gen_session.return_value = "session-uuid-123"

            assign_task(agent, "c-123", state, project_root=project_root)

            assert agent.status == "ready"
            assert agent.task == "c-123"
            assert agent.branch == "agent/test-agent-c-123"
            assert agent.session == "session-uuid-123"
            assert agent.worktree == worktree_path
            assert agent.step_count == 0

    def test_assign_task_creates_claude_md(self, project_root: Path):
        """assign_task creates CLAUDE.md in worktree."""
        state = State()
        agent = spawn_worker("test-agent", state, project_root=project_root)

        with patch("crew.runner.create_worktree") as mock_create_wt, \
             patch("crew.runner.get_task_description") as mock_get_desc, \
             patch("crew.runner.generate_session_id") as mock_gen_session:

            worktree_path = project_root / "agents" / "test-agent-c-456"
            worktree_path.mkdir(parents=True, exist_ok=True)
            mock_create_wt.return_value = worktree_path
            mock_get_desc.return_value = "Build the widget"
            mock_gen_session.return_value = "session-uuid-456"

            assign_task(agent, "c-456", state, project_root=project_root)

            claude_md = worktree_path / "CLAUDE.md"
            assert claude_md.exists()
            content = claude_md.read_text()
            assert "test-agent" in content
            assert "c-456" in content
            assert "Build the widget" in content

    def test_assign_task_fails_for_non_idle_agent(self, project_root: Path):
        """assign_task raises error if agent is not idle."""
        state = State()
        agent = spawn_worker("test-agent", state, project_root=project_root)
        agent.status = "working"

        with pytest.raises(RuntimeError) as exc_info:
            assign_task(agent, "c-123", state, project_root=project_root)

        assert "not idle" in str(exc_info.value)


class TestStepAgent:
    """Tests for step_agent function."""

    def test_step_agent_first_step_uses_init_prompt(self, project_root: Path):
        """step_agent uses INIT_PROMPT for first step."""
        state = State()
        agent = Agent(
            name="test-agent",
            session="session-123",
            worktree=project_root,
            branch="agent/test-branch",
            task="c-100",
            status="ready",
            started_at=datetime.now(),
            step_count=0,
        )
        state.add_agent(agent)

        mock_response = {
            "result": "Starting work...",
            "usage": {"input_tokens": 200, "output_tokens": 100},
            "total_cost_usd": 0.01,
        }

        with patch("crew.runner.run_claude") as mock_run_claude, \
             patch("crew.runner.write_log"):
            mock_run_claude.return_value = {
                "result": mock_response["result"],
                "input_tokens": mock_response["usage"]["input_tokens"],
                "output_tokens": mock_response["usage"]["output_tokens"],
                "cost_usd": mock_response["total_cost_usd"],
                "stderr": "",
            }

            step_agent(agent, state, project_root=project_root)

            call_args = mock_run_claude.call_args
            assert call_args[1]["is_new_session"] is True

    def test_step_agent_subsequent_steps_resume_session(self, project_root: Path):
        """step_agent resumes session for subsequent steps."""
        state = State()
        agent = Agent(
            name="test-agent",
            session="session-123",
            worktree=project_root,
            branch="agent/test-branch",
            task="c-100",
            status="working",
            started_at=datetime.now(),
            step_count=3,
        )
        state.add_agent(agent)

        with patch("crew.runner.run_claude") as mock_run_claude, \
             patch("crew.runner.write_log"):
            mock_run_claude.return_value = {
                "result": "Continuing work...",
                "input_tokens": 150,
                "output_tokens": 80,
                "cost_usd": 0.008,
                "stderr": "",
            }

            step_agent(agent, state, project_root=project_root)

            call_args = mock_run_claude.call_args
            assert call_args[1]["is_new_session"] is False

    def test_step_agent_accumulates_tokens(self, project_root: Path):
        """step_agent accumulates token usage across steps."""
        state = State()
        agent = Agent(
            name="test-agent",
            session="session-123",
            worktree=project_root,
            branch="agent/test-branch",
            task="c-100",
            status="ready",
            started_at=datetime.now(),
            step_count=0,
            total_input_tokens=500,
            total_output_tokens=250,
            total_cost_usd=0.025,
        )
        state.add_agent(agent)

        with patch("crew.runner.run_claude") as mock_run_claude, \
             patch("crew.runner.write_log"):
            mock_run_claude.return_value = {
                "result": "Working...",
                "input_tokens": 200,
                "output_tokens": 100,
                "cost_usd": 0.01,
                "stderr": "",
            }

            step_agent(agent, state, project_root=project_root)

            assert agent.total_input_tokens == 700
            assert agent.total_output_tokens == 350
            assert agent.total_cost_usd == pytest.approx(0.035)

    def test_step_agent_increments_step_count(self, project_root: Path):
        """step_agent increments step count."""
        state = State()
        agent = Agent(
            name="test-agent",
            session="session-123",
            worktree=project_root,
            branch="agent/test-branch",
            task="c-100",
            status="ready",
            started_at=datetime.now(),
            step_count=2,
        )
        state.add_agent(agent)

        with patch("crew.runner.run_claude") as mock_run_claude, \
             patch("crew.runner.write_log"):
            mock_run_claude.return_value = {
                "result": "Working...",
                "input_tokens": 100,
                "output_tokens": 50,
                "cost_usd": 0.005,
                "stderr": "",
            }

            step_agent(agent, state, project_root=project_root)

            assert agent.step_count == 3

    def test_step_agent_detects_done(self, project_root: Path):
        """step_agent sets status to done when output contains DONE."""
        state = State()
        agent = Agent(
            name="test-agent",
            session="session-123",
            worktree=project_root,
            branch="agent/test-branch",
            task="c-100",
            status="working",
            started_at=datetime.now(),
            step_count=5,
        )
        state.add_agent(agent)

        with patch("crew.runner.run_claude") as mock_run_claude, \
             patch("crew.runner.write_log"):
            mock_run_claude.return_value = {
                "result": "Task completed.\nDONE",
                "input_tokens": 100,
                "output_tokens": 50,
                "cost_usd": 0.005,
                "stderr": "",
            }

            step_agent(agent, state, project_root=project_root)

            assert agent.status == "done"

    def test_step_agent_detects_stuck_after_many_steps(self, project_root: Path):
        """step_agent sets status to stuck after 20 steps without completion."""
        state = State()
        agent = Agent(
            name="test-agent",
            session="session-123",
            worktree=project_root,
            branch="agent/test-branch",
            task="c-100",
            status="working",
            started_at=datetime.now(),
            step_count=19,
        )
        state.add_agent(agent)

        with patch("crew.runner.run_claude") as mock_run_claude, \
             patch("crew.runner.write_log"):
            mock_run_claude.return_value = {
                "result": "Still working...",
                "input_tokens": 100,
                "output_tokens": 50,
                "cost_usd": 0.005,
                "stderr": "",
            }

            step_agent(agent, state, project_root=project_root)

            assert agent.step_count == 20
            assert agent.status == "stuck"

    def test_step_agent_updates_last_step_at(self, project_root: Path):
        """step_agent updates last_step_at timestamp."""
        state = State()
        agent = Agent(
            name="test-agent",
            session="session-123",
            worktree=project_root,
            branch="agent/test-branch",
            task="c-100",
            status="ready",
            started_at=datetime.now(),
            step_count=0,
            last_step_at=None,
        )
        state.add_agent(agent)

        with patch("crew.runner.run_claude") as mock_run_claude, \
             patch("crew.runner.write_log"):
            mock_run_claude.return_value = {
                "result": "Working...",
                "input_tokens": 100,
                "output_tokens": 50,
                "cost_usd": 0.005,
                "stderr": "",
            }

            step_agent(agent, state, project_root=project_root)

            assert agent.last_step_at is not None


class TestCompleteTask:
    """Tests for complete_task function."""

    def test_complete_task_closes_ticket(self, project_root: Path):
        """complete_task closes the ticket via tk."""
        state = State()
        agent = Agent(
            name="test-agent",
            session="session-123",
            worktree=project_root / "agents" / "test-worktree",
            branch="agent/test-branch",
            task="c-100",
            status="done",
            started_at=datetime.now(),
            step_count=5,
        )
        state.add_agent(agent)

        with patch("crew.runner.close_ticket") as mock_close, \
             patch("crew.runner.remove_worktree"), \
             patch("crew.runner.run_git"), \
             patch("crew.runner.merge_branch"), \
             patch("crew.runner.delete_branch"):

            complete_task(agent, state, project_root=project_root)

            mock_close.assert_called_once_with("c-100")

    def test_complete_task_removes_worktree(self, project_root: Path):
        """complete_task removes the worktree."""
        state = State()
        worktree = project_root / "agents" / "test-worktree"
        agent = Agent(
            name="test-agent",
            session="session-123",
            worktree=worktree,
            branch="agent/test-branch",
            task="c-100",
            status="done",
            started_at=datetime.now(),
            step_count=5,
        )
        state.add_agent(agent)

        with patch("crew.runner.close_ticket"), \
             patch("crew.runner.remove_worktree") as mock_remove_wt, \
             patch("crew.runner.run_git"), \
             patch("crew.runner.merge_branch"), \
             patch("crew.runner.delete_branch"):

            complete_task(agent, state, project_root=project_root)

            mock_remove_wt.assert_called_once_with(worktree)

    def test_complete_task_merges_branch(self, project_root: Path):
        """complete_task merges the agent's branch."""
        state = State()
        agent = Agent(
            name="test-agent",
            session="session-123",
            worktree=project_root / "agents" / "test-worktree",
            branch="agent/test-branch",
            task="c-100",
            status="done",
            started_at=datetime.now(),
            step_count=5,
        )
        state.add_agent(agent)

        with patch("crew.runner.close_ticket"), \
             patch("crew.runner.remove_worktree"), \
             patch("crew.runner.run_git"), \
             patch("crew.runner.merge_branch") as mock_merge, \
             patch("crew.runner.delete_branch"):

            complete_task(agent, state, project_root=project_root)

            mock_merge.assert_called_once()
            call_args = mock_merge.call_args
            assert "agent/test-branch" in call_args[0]

    def test_complete_task_resets_agent_to_idle(self, project_root: Path):
        """complete_task resets agent to idle state."""
        state = State()
        agent = Agent(
            name="test-agent",
            session="session-123",
            worktree=project_root / "agents" / "test-worktree",
            branch="agent/test-branch",
            task="c-100",
            status="done",
            started_at=datetime.now(),
            step_count=5,
            total_input_tokens=1000,
            total_output_tokens=500,
            total_cost_usd=0.05,
        )
        state.add_agent(agent)

        with patch("crew.runner.close_ticket"), \
             patch("crew.runner.remove_worktree"), \
             patch("crew.runner.run_git"), \
             patch("crew.runner.merge_branch"), \
             patch("crew.runner.delete_branch"):

            complete_task(agent, state, project_root=project_root)

            assert agent.status == "idle"
            assert agent.session == ""
            assert agent.worktree is None
            assert agent.branch == ""
            assert agent.task is None
            assert agent.step_count == 0

    def test_complete_task_deletes_branch(self, project_root: Path):
        """complete_task deletes the merged branch."""
        state = State()
        agent = Agent(
            name="test-agent",
            session="session-123",
            worktree=project_root / "agents" / "test-worktree",
            branch="agent/test-branch",
            task="c-100",
            status="done",
            started_at=datetime.now(),
            step_count=5,
        )
        state.add_agent(agent)

        with patch("crew.runner.close_ticket"), \
             patch("crew.runner.remove_worktree"), \
             patch("crew.runner.run_git"), \
             patch("crew.runner.merge_branch"), \
             patch("crew.runner.delete_branch") as mock_delete:

            complete_task(agent, state, project_root=project_root)

            mock_delete.assert_called_once_with("agent/test-branch")


class TestTokenAccumulation:
    """Tests specifically for token accumulation functionality."""

    def test_multiple_steps_accumulate_tokens(self, project_root: Path):
        """Multiple steps correctly accumulate token usage."""
        state = State()
        agent = Agent(
            name="test-agent",
            session="session-123",
            worktree=project_root,
            branch="agent/test-branch",
            task="c-100",
            status="ready",
            started_at=datetime.now(),
            step_count=0,
            total_input_tokens=0,
            total_output_tokens=0,
            total_cost_usd=0.0,
        )
        state.add_agent(agent)

        step_responses = [
            {"result": "Step 1", "input_tokens": 100, "output_tokens": 50, "cost_usd": 0.005, "stderr": ""},
            {"result": "Step 2", "input_tokens": 150, "output_tokens": 75, "cost_usd": 0.0075, "stderr": ""},
            {"result": "Step 3\nDONE", "input_tokens": 80, "output_tokens": 40, "cost_usd": 0.004, "stderr": ""},
        ]

        with patch("crew.runner.run_claude") as mock_run_claude, \
             patch("crew.runner.write_log"):
            for response in step_responses:
                mock_run_claude.return_value = response
                step_agent(agent, state, project_root=project_root)

            # Total should be sum of all steps
            assert agent.total_input_tokens == 330  # 100 + 150 + 80
            assert agent.total_output_tokens == 165  # 50 + 75 + 40
            assert agent.total_cost_usd == pytest.approx(0.0165)  # 0.005 + 0.0075 + 0.004

    def test_token_accumulation_with_zero_usage(self, project_root: Path):
        """Token accumulation handles zero usage responses."""
        state = State()
        agent = Agent(
            name="test-agent",
            session="session-123",
            worktree=project_root,
            branch="agent/test-branch",
            task="c-100",
            status="ready",
            started_at=datetime.now(),
            step_count=0,
            total_input_tokens=100,
            total_output_tokens=50,
            total_cost_usd=0.01,
        )
        state.add_agent(agent)

        with patch("crew.runner.run_claude") as mock_run_claude, \
             patch("crew.runner.write_log"):
            mock_run_claude.return_value = {
                "result": "No usage reported",
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "stderr": "",
            }

            step_agent(agent, state, project_root=project_root)

            # Totals should remain unchanged
            assert agent.total_input_tokens == 100
            assert agent.total_output_tokens == 50
            assert agent.total_cost_usd == pytest.approx(0.01)


class TestFindClaudeProcess:
    """Tests for find_claude_process function."""

    def test_find_process_by_session_id(self):
        """find_claude_process finds process by session ID."""
        agent = Agent(
            name="test",
            session="abc-123-def",
            worktree=Path("/tmp/test"),
            branch="test-branch",
        )

        ps_output = """PID COMMAND
1234 claude --print --session-id abc-123-def -p prompt
5678 node something else
"""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=ps_output, stderr="")
            pid = find_claude_process(agent)

        assert pid == 1234

    def test_find_process_by_worktree_path(self):
        """find_claude_process finds process by worktree path."""
        agent = Agent(
            name="test",
            session="different-session",
            worktree=Path("/tmp/agents/myworktree"),
            branch="test-branch",
        )

        ps_output = """PID COMMAND
1234 claude --print /tmp/agents/myworktree -p prompt
5678 node something else
"""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=ps_output, stderr="")
            pid = find_claude_process(agent)

        assert pid == 1234

    def test_find_process_returns_none_when_not_found(self):
        """find_claude_process returns None when no matching process."""
        agent = Agent(
            name="test",
            session="no-match",
            worktree=Path("/tmp/no-match"),
            branch="test-branch",
        )

        ps_output = """PID COMMAND
1234 claude --print --session-id other-session -p prompt
5678 node something else
"""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=ps_output, stderr="")
            pid = find_claude_process(agent)

        assert pid is None

    def test_find_process_handles_exception(self):
        """find_claude_process returns None on exception."""
        agent = Agent(
            name="test",
            session="test-session",
            worktree=Path("/tmp/test"),
            branch="test-branch",
        )

        with patch("subprocess.run", side_effect=Exception("ps failed")):
            pid = find_claude_process(agent)

        assert pid is None


class TestKillClaudeProcess:
    """Tests for kill_claude_process function."""

    def test_kill_process_success(self):
        """kill_claude_process returns True when process is killed."""
        agent = Agent(
            name="test",
            session="test-session",
            worktree=Path("/tmp/test"),
            branch="test-branch",
        )

        with patch("crew.runner.find_claude_process", return_value=1234):
            with patch("os.kill") as mock_kill:
                result = kill_claude_process(agent)

        assert result is True
        mock_kill.assert_called_once()

    def test_kill_process_not_found(self):
        """kill_claude_process returns False when no process found."""
        agent = Agent(
            name="test",
            session="test-session",
            worktree=Path("/tmp/test"),
            branch="test-branch",
        )

        with patch("crew.runner.find_claude_process", return_value=None):
            result = kill_claude_process(agent)

        assert result is False

    def test_kill_process_already_gone(self):
        """kill_claude_process returns False when process already gone."""
        agent = Agent(
            name="test",
            session="test-session",
            worktree=Path("/tmp/test"),
            branch="test-branch",
        )

        with patch("crew.runner.find_claude_process", return_value=1234):
            with patch("os.kill", side_effect=ProcessLookupError):
                result = kill_claude_process(agent)

        assert result is False

    def test_kill_process_permission_error(self):
        """kill_claude_process returns False on PermissionError."""
        agent = Agent(
            name="test",
            session="test-session",
            worktree=Path("/tmp/test"),
            branch="test-branch",
        )

        with patch("crew.runner.find_claude_process", return_value=1234):
            with patch("os.kill", side_effect=PermissionError):
                result = kill_claude_process(agent)

        assert result is False

    def test_kill_process_uses_sigterm(self):
        """kill_claude_process sends SIGTERM signal."""
        import signal

        agent = Agent(
            name="test",
            session="test-session",
            worktree=Path("/tmp/test"),
            branch="test-branch",
        )

        with patch("crew.runner.find_claude_process", return_value=1234):
            with patch("os.kill") as mock_kill:
                kill_claude_process(agent)

        mock_kill.assert_called_once_with(1234, signal.SIGTERM)


class TestCheckWorkCompleted:
    """Tests for check_work_completed function."""

    def test_returns_done_when_done_in_logs(self, project_root: Path):
        """check_work_completed returns 'done' when DONE found in logs."""
        agent = Agent(
            name="test-agent",
            session="test-session",
            worktree=Path("/tmp/test"),
            branch="test-branch",
            step_count=5,
        )

        log_content = """=== CREW LOG ===
Agent: test-agent
Time: 2026-01-01T12:00:00
Step: 5
Prompt: Continue
Session: test-session
---
I have completed the task.

DONE
=== END ===
"""
        with patch("crew.runner.read_latest_log", return_value=log_content):
            result = check_work_completed(agent, project_root)

        assert result == "done"

    def test_returns_partial_when_steps_but_no_done(self, project_root: Path):
        """check_work_completed returns 'partial' when steps taken but no DONE."""
        agent = Agent(
            name="test-agent",
            session="test-session",
            worktree=Path("/tmp/test"),
            branch="test-branch",
            step_count=3,
        )

        log_content = """=== CREW LOG ===
Agent: test-agent
Time: 2026-01-01T12:00:00
Step: 3
Prompt: Continue
Session: test-session
---
Still working on this...
=== END ===
"""
        with patch("crew.runner.read_latest_log", return_value=log_content):
            result = check_work_completed(agent, project_root)

        assert result == "partial"

    def test_returns_nothing_when_no_logs(self, project_root: Path):
        """check_work_completed returns 'nothing' when no logs found."""
        agent = Agent(
            name="test-agent",
            session="test-session",
            worktree=Path("/tmp/test"),
            branch="test-branch",
            step_count=0,
        )

        with patch("crew.runner.read_latest_log", return_value=None):
            result = check_work_completed(agent, project_root)

        assert result == "nothing"

    def test_returns_nothing_when_empty_logs(self, project_root: Path):
        """check_work_completed returns 'nothing' when log content is empty."""
        agent = Agent(
            name="test-agent",
            session="test-session",
            worktree=Path("/tmp/test"),
            branch="test-branch",
            step_count=0,
        )

        with patch("crew.runner.read_latest_log", return_value=""):
            result = check_work_completed(agent, project_root)

        assert result == "nothing"

    def test_returns_nothing_when_zero_steps(self, project_root: Path):
        """check_work_completed returns 'nothing' when step_count is 0 without DONE."""
        agent = Agent(
            name="test-agent",
            session="test-session",
            worktree=Path("/tmp/test"),
            branch="test-branch",
            step_count=0,
        )

        log_content = """=== CREW LOG ===
Agent: test-agent
---
Some output but no DONE marker
=== END ===
"""
        with patch("crew.runner.read_latest_log", return_value=log_content):
            result = check_work_completed(agent, project_root)

        assert result == "nothing"

    def test_detects_done_in_output_section(self, project_root: Path):
        """check_work_completed correctly parses DONE from output section."""
        agent = Agent(
            name="test-agent",
            session="test-session",
            worktree=Path("/tmp/test"),
            branch="test-branch",
            step_count=3,
        )

        # DONE appears after the --- separator (in output section)
        log_content = """=== CREW LOG ===
Agent: test-agent
Time: 2026-01-01T12:00:00
Step: 3
Prompt: Continue
Session: test-session
---
Finished the implementation.

DONE
=== END ===
"""
        with patch("crew.runner.read_latest_log", return_value=log_content):
            result = check_work_completed(agent, project_root)

        assert result == "done"

    def test_done_case_insensitive_in_logs(self, project_root: Path):
        """check_work_completed detects DONE in any case."""
        agent = Agent(
            name="test-agent",
            session="test-session",
            worktree=Path("/tmp/test"),
            branch="test-branch",
            step_count=2,
        )

        log_content = """=== CREW LOG ===
---
done
=== END ===
"""
        with patch("crew.runner.read_latest_log", return_value=log_content):
            result = check_work_completed(agent, project_root)

        assert result == "done"

    def test_embedded_done_not_detected(self, project_root: Path):
        """check_work_completed does not detect DONE embedded in other words."""
        agent = Agent(
            name="test-agent",
            session="test-session",
            worktree=Path("/tmp/test"),
            branch="test-branch",
            step_count=2,
        )

        log_content = """=== CREW LOG ===
---
UNDONE task and more work to do
=== END ===
"""
        with patch("crew.runner.read_latest_log", return_value=log_content):
            result = check_work_completed(agent, project_root)

        assert result == "partial"


class TestShutdownAgent:
    """Tests for shutdown_agent function."""

    def test_shutdown_marks_done_when_work_completed(self, project_root: Path):
        """shutdown_agent sets status to 'done' when work is complete."""
        agent = Agent(
            name="test-agent",
            session="test-session",
            worktree=Path("/tmp/test"),
            branch="test-branch",
            status="working",
            step_count=5,
        )
        state = State()
        state.add_agent(agent)

        with patch("crew.runner.kill_claude_process", return_value=True):
            with patch("crew.runner.check_work_completed", return_value="done"):
                result = shutdown_agent(agent, state, project_root)

        assert result == "done"
        assert agent.status == "done"

    def test_shutdown_keeps_working_when_partial(self, project_root: Path):
        """shutdown_agent keeps status as 'working' for partial work."""
        agent = Agent(
            name="test-agent",
            session="test-session",
            worktree=Path("/tmp/test"),
            branch="test-branch",
            status="working",
            step_count=3,
        )
        state = State()
        state.add_agent(agent)

        with patch("crew.runner.kill_claude_process", return_value=True):
            with patch("crew.runner.check_work_completed", return_value="partial"):
                result = shutdown_agent(agent, state, project_root)

        assert result == "partial"
        assert agent.status == "working"

    def test_shutdown_resets_to_ready_when_nothing(self, project_root: Path):
        """shutdown_agent resets status to 'ready' when no work done."""
        agent = Agent(
            name="test-agent",
            session="test-session",
            worktree=Path("/tmp/test"),
            branch="test-branch",
            status="working",
            step_count=0,
        )
        state = State()
        state.add_agent(agent)

        with patch("crew.runner.kill_claude_process", return_value=True):
            with patch("crew.runner.check_work_completed", return_value="nothing"):
                result = shutdown_agent(agent, state, project_root)

        assert result == "nothing"
        assert agent.status == "ready"

    def test_shutdown_saves_state(self, project_root: Path):
        """shutdown_agent saves state after reconciliation."""
        agent = Agent(
            name="test-agent",
            session="test-session",
            worktree=Path("/tmp/test"),
            branch="test-branch",
            status="working",
            step_count=5,
        )
        state = State()
        state.add_agent(agent)

        with patch("crew.runner.kill_claude_process", return_value=True):
            with patch("crew.runner.check_work_completed", return_value="done"):
                shutdown_agent(agent, state, project_root)

        # Reload state and verify it was saved
        loaded_state = load_state(project_root)
        loaded_agent = loaded_state.get_agent("test-agent")
        assert loaded_agent is not None
        assert loaded_agent.status == "done"

    def test_shutdown_works_without_process(self, project_root: Path):
        """shutdown_agent works even if no process is running."""
        agent = Agent(
            name="test-agent",
            session="test-session",
            worktree=Path("/tmp/test"),
            branch="test-branch",
            status="working",
            step_count=3,
        )
        state = State()
        state.add_agent(agent)

        with patch("crew.runner.kill_claude_process", return_value=False):
            with patch("crew.runner.check_work_completed", return_value="partial"):
                result = shutdown_agent(agent, state, project_root)

        assert result == "partial"
        assert agent.status == "working"

    def test_shutdown_does_not_change_stuck_status_on_partial(self, project_root: Path):
        """shutdown_agent preserves 'stuck' status for partial work."""
        agent = Agent(
            name="test-agent",
            session="test-session",
            worktree=Path("/tmp/test"),
            branch="test-branch",
            status="stuck",
            step_count=20,
        )
        state = State()
        state.add_agent(agent)

        with patch("crew.runner.kill_claude_process", return_value=True):
            with patch("crew.runner.check_work_completed", return_value="partial"):
                result = shutdown_agent(agent, state, project_root)

        assert result == "partial"
        # stuck status should be preserved
        assert agent.status == "stuck"

    def test_shutdown_does_not_change_done_status_on_partial(self, project_root: Path):
        """shutdown_agent preserves 'done' status even if log shows partial."""
        agent = Agent(
            name="test-agent",
            session="test-session",
            worktree=Path("/tmp/test"),
            branch="test-branch",
            status="done",
            step_count=8,
        )
        state = State()
        state.add_agent(agent)

        with patch("crew.runner.kill_claude_process", return_value=True):
            with patch("crew.runner.check_work_completed", return_value="partial"):
                result = shutdown_agent(agent, state, project_root)

        assert result == "partial"
        # done status should be preserved
        assert agent.status == "done"

    def test_shutdown_transitions_ready_to_working_on_partial(self, project_root: Path):
        """shutdown_agent transitions 'ready' to 'working' on partial work."""
        agent = Agent(
            name="test-agent",
            session="test-session",
            worktree=Path("/tmp/test"),
            branch="test-branch",
            status="ready",
            step_count=1,
        )
        state = State()
        state.add_agent(agent)

        with patch("crew.runner.kill_claude_process", return_value=True):
            with patch("crew.runner.check_work_completed", return_value="partial"):
                result = shutdown_agent(agent, state, project_root)

        assert result == "partial"
        assert agent.status == "working"

    def test_shutdown_transitions_idle_to_working_on_partial(self, project_root: Path):
        """shutdown_agent transitions 'idle' to 'working' on partial work."""
        agent = Agent(
            name="test-agent",
            session="test-session",
            worktree=Path("/tmp/test"),
            branch="test-branch",
            status="idle",
            step_count=2,
        )
        state = State()
        state.add_agent(agent)

        with patch("crew.runner.kill_claude_process", return_value=True):
            with patch("crew.runner.check_work_completed", return_value="partial"):
                result = shutdown_agent(agent, state, project_root)

        assert result == "partial"
        assert agent.status == "working"

    def test_shutdown_calls_kill_before_check_work(self, project_root: Path):
        """shutdown_agent kills process before checking work status."""
        agent = Agent(
            name="test-agent",
            session="test-session",
            worktree=Path("/tmp/test"),
            branch="test-branch",
            status="working",
            step_count=3,
        )
        state = State()
        state.add_agent(agent)

        call_order = []

        def mock_kill(a):
            call_order.append("kill")
            return True

        def mock_check(a, p):
            call_order.append("check")
            return "partial"

        with patch("crew.runner.kill_claude_process", side_effect=mock_kill):
            with patch("crew.runner.check_work_completed", side_effect=mock_check):
                shutdown_agent(agent, state, project_root)

        assert call_order == ["kill", "check"]

    def test_shutdown_returns_work_status(self, project_root: Path):
        """shutdown_agent returns the work completion status."""
        agent = Agent(
            name="test-agent",
            session="test-session",
            worktree=Path("/tmp/test"),
            branch="test-branch",
            status="working",
            step_count=5,
        )
        state = State()
        state.add_agent(agent)

        for expected_status in ["done", "partial", "nothing"]:
            agent.status = "working"  # Reset for each test
            with patch("crew.runner.kill_claude_process", return_value=True):
                with patch("crew.runner.check_work_completed", return_value=expected_status):
                    result = shutdown_agent(agent, state, project_root)

            assert result == expected_status

    def test_shutdown_uses_project_root_for_state(self, project_root: Path):
        """shutdown_agent uses project_root for saving state."""
        agent = Agent(
            name="test-agent",
            session="test-session",
            worktree=Path("/tmp/test"),
            branch="test-branch",
            status="working",
            step_count=5,
        )
        state = State()
        state.add_agent(agent)

        with patch("crew.runner.kill_claude_process", return_value=True):
            with patch("crew.runner.check_work_completed", return_value="done"):
                shutdown_agent(agent, state, project_root)

        # Verify state was saved to project_root
        loaded_state = load_state(project_root)
        assert loaded_state.get_agent("test-agent") is not None

    def test_shutdown_uses_project_root_for_check_work(self, project_root: Path):
        """shutdown_agent passes project_root to check_work_completed."""
        agent = Agent(
            name="test-agent",
            session="test-session",
            worktree=Path("/tmp/test"),
            branch="test-branch",
            status="working",
            step_count=5,
        )
        state = State()
        state.add_agent(agent)

        with patch("crew.runner.kill_claude_process", return_value=True):
            with patch("crew.runner.check_work_completed", return_value="done") as mock_check:
                shutdown_agent(agent, state, project_root)

        mock_check.assert_called_once_with(agent, project_root)
