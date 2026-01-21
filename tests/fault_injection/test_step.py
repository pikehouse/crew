"""Fault injection tests for step_agent operation.

Tests interrupt scenarios during agent stepping:
1. Kill before run_claude → agent still "working", can retry
2. Timeout during run_claude → worktree has partial work, RESUME_DIRTY_PROMPT used
3. Kill after step_count increment → step counted, can continue
4. Kill after "done" detected → agent is "done", can complete

Key invariant: Worktree preserves work even if step is interrupted.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from crew.agent import Agent
from crew.runner import (
    spawn_worker,
    assign_task,
    step_agent,
    is_done,
    INIT_PROMPT,
    STEP_PROMPT,
    RESUME_DIRTY_PROMPT,
)
from crew.state import State, save_state, load_state
from tests.fault_injection.invariants import assert_invariants, check_all_invariants
from tests.fixtures.mock_claude import MockClaude, mock_done_on_step


class TestStepAgentInterrupts:
    """Tests for interrupts during step_agent."""

    def _setup_ready_agent(self, git_project: Path) -> tuple[Agent, State]:
        """Helper to create a ready agent for stepping."""
        state = State()
        agent = spawn_worker("test-agent", state, git_project)

        with patch("crew.runner.get_task_description") as mock_desc:
            mock_desc.return_value = "Test task"
            assign_task(agent, "t-0001", state, git_project)

        return agent, state

    def test_step_changes_status_to_working(self, git_project: Path):
        """Baseline: step_agent changes status from ready to working."""
        agent, state = self._setup_ready_agent(git_project)

        mock = MockClaude()
        mock.configure("success", responses=["Working on task..."])

        with patch("crew.runner.run_claude", mock):
            with patch("crew.runner._refresh_agent_summary"):
                step_agent(agent, state, project_root=git_project)

        assert agent.status == "working"
        assert agent.step_count == 1
        assert_invariants(state, git_project)

    def test_step_detects_done_marker(self, git_project: Path):
        """step_agent detects DONE marker and sets status to done."""
        agent, state = self._setup_ready_agent(git_project)

        mock = MockClaude()
        mock.configure("success", responses=["Task complete!\n\nDONE"])

        with patch("crew.runner.run_claude", mock):
            with patch("crew.runner._refresh_agent_summary"):
                result = step_agent(agent, state, project_root=git_project)

        assert "DONE" in result
        assert agent.status == "done"
        assert_invariants(state, git_project)

    def test_timeout_during_step_preserves_state(self, git_project: Path):
        """Timeout during run_claude preserves working state."""
        agent, state = self._setup_ready_agent(git_project)

        mock = MockClaude()
        mock.configure("timeout")

        with patch("crew.runner.run_claude", mock):
            with pytest.raises(TimeoutError):
                step_agent(agent, state, project_root=git_project)

        # Agent should be in working state (status set before run_claude)
        reloaded = load_state(git_project)
        reloaded_agent = reloaded.get_agent("test-agent")
        assert reloaded_agent.status == "working"

        # Step count should not have incremented (timeout before increment)
        assert reloaded_agent.step_count == 0

        # Invariants should hold
        assert_invariants(reloaded, git_project)

    def test_retry_after_timeout_uses_resume_prompt(self, git_project: Path):
        """After timeout, retry with step_count=0 but dirty worktree uses RESUME_DIRTY_PROMPT."""
        agent, state = self._setup_ready_agent(git_project)

        # First step times out
        mock = MockClaude()
        mock.configure("timeout")

        with patch("crew.runner.run_claude", mock):
            with pytest.raises(TimeoutError):
                step_agent(agent, state, project_root=git_project)

        # Simulate partial work in worktree (uncommitted changes)
        test_file = agent.worktree / "partial_work.py"
        test_file.write_text("# Partial work from interrupted step\n")

        # Git add to make it tracked but uncommitted
        subprocess.run(
            ["git", "add", "partial_work.py"],
            cwd=agent.worktree,
            capture_output=True,
        )

        # Reload state
        state2 = load_state(git_project)
        agent2 = state2.get_agent("test-agent")

        # Mock that captures the prompt
        captured_prompts = []

        def capture_prompt_mock(*args, **kwargs):
            if args:
                captured_prompts.append(args[0])
            return {
                "result": "Continuing work...",
                "input_tokens": 100,
                "output_tokens": 150,
                "cost_usd": 0.001,
                "stderr": "",
            }

        with patch("crew.runner.run_claude", side_effect=capture_prompt_mock):
            with patch("crew.runner._refresh_agent_summary"):
                step_agent(agent2, state2, project_root=git_project)

        # Should have used RESUME_DIRTY_PROMPT since worktree has changes
        assert len(captured_prompts) == 1
        assert "previous session made progress" in captured_prompts[0].lower() or \
               "uncommitted changes" in captured_prompts[0].lower() or \
               "continue from where" in captured_prompts[0].lower()

    def test_interrupt_after_step_count_increment(self, git_project: Path):
        """If interrupted after step_count increment, step is counted."""
        agent, state = self._setup_ready_agent(git_project)

        save_called = {"count": 0}
        original_save = save_state

        def save_that_tracks(s, p):
            save_called["count"] += 1
            # Interrupt after second save (which happens after step_count increment)
            if save_called["count"] == 2:
                original_save(s, p)  # Save successfully first
                raise RuntimeError("Simulated interrupt")
            return original_save(s, p)

        mock = MockClaude()
        mock.configure("success", responses=["Working..."])

        with patch("crew.runner.run_claude", mock):
            with patch("crew.runner._refresh_agent_summary"):
                with patch("crew.runner.save_state", side_effect=save_that_tracks):
                    with pytest.raises(RuntimeError, match="Simulated interrupt"):
                        step_agent(agent, state, project_root=git_project)

        # State should have been saved with step_count = 1
        reloaded = load_state(git_project)
        reloaded_agent = reloaded.get_agent("test-agent")
        assert reloaded_agent.step_count == 1
        assert reloaded_agent.status == "working"

    def test_session_conflict_auto_recovery(self, git_project: Path):
        """Session conflict error triggers session regeneration and retry."""
        agent, state = self._setup_ready_agent(git_project)

        call_count = {"count": 0}

        def mock_with_session_conflict(*args, **kwargs):
            call_count["count"] += 1
            if call_count["count"] == 1:
                # First call: session conflict
                return {
                    "result": "",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                    "stderr": "Error: Session ID already in use",
                }
            else:
                # Second call: success
                return {
                    "result": "Working on task...",
                    "input_tokens": 100,
                    "output_tokens": 150,
                    "cost_usd": 0.001,
                    "stderr": "",
                }

        original_session = agent.session

        with patch("crew.runner.run_claude", side_effect=mock_with_session_conflict):
            with patch("crew.runner._refresh_agent_summary"):
                step_agent(agent, state, project_root=git_project)

        # Session should have been regenerated
        assert agent.session != original_session
        assert agent.status == "working"
        assert_invariants(state, git_project)

    def test_multiple_steps_accumulate_tokens(self, git_project: Path):
        """Multiple steps accumulate token counts and costs."""
        agent, state = self._setup_ready_agent(git_project)

        responses = ["Step 1 work...", "Step 2 work...", "Final step\n\nDONE"]

        for i, response in enumerate(responses):
            mock = MockClaude()
            mock.configure("success", responses=[response])

            with patch("crew.runner.run_claude", mock):
                with patch("crew.runner._refresh_agent_summary"):
                    step_agent(agent, state, project_root=git_project)

            if "DONE" in response:
                break

        assert agent.step_count == 3
        assert agent.total_input_tokens > 0
        assert agent.total_output_tokens > 0
        assert agent.total_cost_usd > 0
        assert agent.status == "done"
        assert_invariants(state, git_project)


class TestStepAgentIdempotency:
    """Tests verifying idempotency of step_agent operations."""

    def _setup_ready_agent(self, git_project: Path) -> tuple[Agent, State]:
        """Helper to create a ready agent for stepping."""
        state = State()
        agent = spawn_worker("test-agent", state, git_project)

        with patch("crew.runner.get_task_description") as mock_desc:
            mock_desc.return_value = "Test task"
            assign_task(agent, "t-0001", state, git_project)

        return agent, state

    def test_step_can_be_retried_after_timeout(self, git_project: Path):
        """Step can be retried after timeout and will succeed."""
        agent, state = self._setup_ready_agent(git_project)

        # First attempt times out
        mock1 = MockClaude()
        mock1.configure("timeout")

        with patch("crew.runner.run_claude", mock1):
            with pytest.raises(TimeoutError):
                step_agent(agent, state, project_root=git_project)

        # Retry succeeds
        mock2 = MockClaude()
        mock2.configure("success", responses=["Working on task..."])

        with patch("crew.runner.run_claude", mock2):
            with patch("crew.runner._refresh_agent_summary"):
                result = step_agent(agent, state, project_root=git_project)

        assert "Working" in result
        assert agent.status == "working"
        assert agent.step_count == 1
        assert_invariants(state, git_project)

    def test_resume_preserves_progress(self, git_project: Path):
        """Resuming after interrupt preserves progress in worktree."""
        agent, state = self._setup_ready_agent(git_project)

        # First step succeeds and makes progress
        mock1 = MockClaude()
        mock1.configure("success", responses=["Created feature.py"])

        with patch("crew.runner.run_claude", mock1):
            with patch("crew.runner._refresh_agent_summary"):
                step_agent(agent, state, project_root=git_project)

        # Simulate work being done in worktree
        feature_file = agent.worktree / "feature.py"
        feature_file.write_text("def new_feature():\n    pass\n")
        subprocess.run(
            ["git", "add", "feature.py"],
            cwd=agent.worktree,
            capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Add feature"],
            cwd=agent.worktree,
            capture_output=True,
        )

        # Second step times out
        mock2 = MockClaude()
        mock2.configure("timeout")

        with patch("crew.runner.run_claude", mock2):
            with pytest.raises(TimeoutError):
                step_agent(agent, state, project_root=git_project)

        # Feature file should still exist
        assert feature_file.exists()

        # Third step succeeds and completes
        mock3 = MockClaude()
        mock3.configure("success", responses=["DONE"])

        with patch("crew.runner.run_claude", mock3):
            with patch("crew.runner._refresh_agent_summary"):
                step_agent(agent, state, project_root=git_project)

        assert agent.status == "done"
        assert feature_file.exists()  # Work preserved
        assert_invariants(state, git_project)


class TestIsDoneDetection:
    """Tests for DONE marker detection."""

    def test_done_on_own_line(self):
        """DONE on its own line is detected."""
        assert is_done("Some output\n\nDONE\n")
        assert is_done("DONE")
        assert is_done("Work complete.\nDONE")

    def test_done_case_insensitive(self):
        """DONE detection is case insensitive."""
        assert is_done("done")
        assert is_done("Done")
        assert is_done("DONE")

    def test_done_not_in_middle_of_text(self):
        """DONE embedded in other text is not detected."""
        assert not is_done("I have done the work")
        assert not is_done("UNDONE task")
        assert not is_done("The task is done but more work needed")

    def test_done_with_whitespace(self):
        """DONE with surrounding whitespace is detected."""
        assert is_done("  DONE  ")
        assert is_done("\nDONE\n")
        assert is_done("Output\n\n  DONE  \n\n")
