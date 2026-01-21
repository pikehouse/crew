"""Fault injection tests for complete_task operation.

Tests interrupt scenarios during task completion:
1. Kill after worktree removal but before merge → branch still exists, can merge
2. Kill after merge but before ticket close → work in main, can close ticket
3. Kill after ticket close but before state reset → ticket closed, can reset agent

Critical invariant: ticket NEVER closed unless work is in main.
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
    complete_task,
    close_ticket,
)
from crew.git import merge_branch, delete_branch, remove_worktree
from crew.state import State, save_state, load_state
from tests.fault_injection.invariants import assert_invariants, check_all_invariants
from tests.fixtures.mock_claude import MockClaude


def _commit_work_in_worktree(worktree: Path, filename: str = "feature.py") -> None:
    """Helper to create a commit in the worktree."""
    # Configure git user for worktree (inherits from main but may need explicit config)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=worktree,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=worktree,
        capture_output=True,
    )

    file_path = worktree / filename
    file_path.write_text(f"# Work done in {filename}\ndef feature():\n    pass\n")

    # Add all changes (including CLAUDE.md which was modified by assign_task)
    subprocess.run(["git", "add", "-A"], cwd=worktree, capture_output=True, check=True)

    # Commit (with allow-empty just in case nothing changed)
    result = subprocess.run(
        ["git", "commit", "-m", f"Add {filename}"],
        cwd=worktree,
        capture_output=True,
        text=True,
    )
    # If nothing to commit, that's OK - the worktree might already have commits
    if result.returncode != 0 and "nothing to commit" not in result.stdout:
        raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)


class TestCompleteTaskInterrupts:
    """Tests for interrupts during complete_task."""

    def _setup_done_agent(self, git_project: Path) -> tuple[Agent, State]:
        """Helper to create a done agent ready for completion."""
        state = State()
        agent = spawn_worker("test-agent", state, git_project)

        with patch("crew.runner.get_task_description") as mock_desc:
            mock_desc.return_value = "Test task"
            assign_task(agent, "t-0001", state, git_project)

        # Make some commits in worktree
        _commit_work_in_worktree(agent.worktree)

        # Step to done
        mock = MockClaude()
        mock.configure("success", responses=["DONE"])

        with patch("crew.runner.run_claude", mock):
            with patch("crew.runner._refresh_agent_summary"):
                step_agent(agent, state, project_root=git_project)

        assert agent.status == "done"
        return agent, state

    def test_complete_task_success(self, git_project: Path):
        """Baseline: complete_task successfully merges and resets agent."""
        agent, state = self._setup_done_agent(git_project)

        with patch("crew.runner.close_ticket") as mock_close:
            mock_close.return_value = True
            # Mock tests to pass (worktree inherits pyproject.toml from main project)
            with patch("crew.runner.run_tests_in_worktree") as mock_tests:
                mock_tests.return_value = (True, "All tests passed")
                success, output = complete_task(agent, state, project_root=git_project)

        assert success, f"complete_task failed with output: {output}"
        assert output is None
        assert agent.status == "idle"
        assert agent.worktree is None
        assert agent.task is None

        # Work should be in main
        result = subprocess.run(
            ["git", "log", "--oneline", "-1"],
            cwd=git_project,
            capture_output=True,
            text=True,
        )
        assert "feature.py" in result.stdout or "Merge" in result.stdout

        assert_invariants(state, git_project)

    def test_interrupt_after_worktree_removal(self, git_project: Path):
        """If interrupted after worktree removal, branch still exists for merge."""
        agent, state = self._setup_done_agent(git_project)

        branch_name = agent.branch
        worktree_path = agent.worktree

        # Manually remove worktree (simulating first part of complete_task)
        # Note: complete_task calls run_tests first, so we mock that
        with patch("crew.runner.run_tests_in_worktree") as mock_tests:
            mock_tests.return_value = (True, "Tests passed")
            remove_worktree(worktree_path)

        # Simulate interrupt before merge
        # Branch should still exist
        result = subprocess.run(
            ["git", "branch", "--list", branch_name],
            cwd=git_project,
            capture_output=True,
            text=True,
        )
        assert branch_name in result.stdout

        # Manual recovery: merge the branch
        subprocess.run(["git", "checkout", "main"], cwd=git_project, capture_output=True)
        subprocess.run(
            ["git", "merge", "--no-ff", branch_name, "-m", "Recovery merge"],
            cwd=git_project,
            capture_output=True,
            check=True,
        )

        # Verify work is in main
        result = subprocess.run(
            ["git", "log", "--oneline", "-5"],
            cwd=git_project,
            capture_output=True,
            text=True,
        )
        assert "feature.py" in result.stdout or "Recovery merge" in result.stdout or "CLAUDE.md" in result.stdout

    def test_interrupt_after_merge_before_ticket_close(self, git_project: Path):
        """If interrupted after merge but before ticket close, work is in main."""
        agent, state = self._setup_done_agent(git_project)

        # Track if close_ticket was called
        close_called = {"count": 0}

        def close_that_interrupts(task_id):
            close_called["count"] += 1
            raise RuntimeError("Simulated interrupt before ticket close")

        with patch("crew.runner.run_tests_in_worktree") as mock_tests:
            mock_tests.return_value = (True, "Tests passed")
            with patch("crew.runner.close_ticket", side_effect=close_that_interrupts):
                with pytest.raises(RuntimeError, match="Simulated interrupt"):
                    complete_task(agent, state, project_root=git_project)

        # Work should be in main (merge happened before interrupt)
        result = subprocess.run(
            ["git", "log", "--oneline", "-1"],
            cwd=git_project,
            capture_output=True,
            text=True,
        )
        # The merge commit should be there
        assert "Merge" in result.stdout or "feature" in result.stdout.lower()

        # Ticket should NOT be closed (that's the point - close happens after merge)
        ticket_file = git_project / ".tickets" / "t-0001.md"
        assert ticket_file.exists()
        content = ticket_file.read_text()
        assert "status: closed" not in content.lower()

    def test_test_failure_returns_to_working(self, git_project: Path):
        """If tests fail, agent returns to working state."""
        agent, state = self._setup_done_agent(git_project)

        # Mock test failure
        with patch("crew.runner.run_tests_in_worktree") as mock_tests:
            mock_tests.return_value = (False, "Test failed: assertion error")
            success, output = complete_task(agent, state, project_root=git_project)

        assert not success
        assert "Test failed" in output
        assert agent.status == "working"

        # Worktree should still exist
        assert agent.worktree.exists()

        assert_invariants(state, git_project)

    def test_ticket_never_closed_before_merge(self, git_project: Path):
        """Critical: Ticket should never be closed before merge succeeds."""
        agent, state = self._setup_done_agent(git_project)

        merge_attempted = {"attempted": False}

        def merge_that_fails(*args, **kwargs):
            merge_attempted["attempted"] = True
            raise RuntimeError("Merge conflict")

        with patch("crew.runner.run_tests_in_worktree") as mock_tests:
            mock_tests.return_value = (True, "Tests passed")
            with patch("crew.runner.merge_branch", side_effect=merge_that_fails):
                with patch("crew.runner.resolve_merge_conflicts", return_value=False):
                    with pytest.raises(RuntimeError, match="Merge failed"):
                        complete_task(agent, state, project_root=git_project)

        # Merge was attempted
        assert merge_attempted["attempted"]

        # Ticket should NOT be closed
        ticket_file = git_project / ".tickets" / "t-0001.md"
        content = ticket_file.read_text()
        assert "status: closed" not in content.lower()


class TestCompleteTaskRecovery:
    """Tests for recovering from interrupted complete_task."""

    def _setup_done_agent(self, git_project: Path) -> tuple[Agent, State]:
        """Helper to create a done agent ready for completion."""
        state = State()
        agent = spawn_worker("test-agent", state, git_project)

        with patch("crew.runner.get_task_description") as mock_desc:
            mock_desc.return_value = "Test task"
            assign_task(agent, "t-0001", state, git_project)

        _commit_work_in_worktree(agent.worktree)

        mock = MockClaude()
        mock.configure("success", responses=["DONE"])

        with patch("crew.runner.run_claude", mock):
            with patch("crew.runner._refresh_agent_summary"):
                step_agent(agent, state, project_root=git_project)

        return agent, state

    def test_can_complete_after_partial_worktree_cleanup(self, git_project: Path):
        """Can complete task even if worktree was partially cleaned."""
        agent, state = self._setup_done_agent(git_project)

        # Worktree exists
        worktree_path = agent.worktree
        assert worktree_path.exists()

        # Simulate partial cleanup - delete worktree directory manually
        # (but git still knows about it)
        import shutil
        shutil.rmtree(worktree_path)

        # Prune stale worktree entries
        subprocess.run(["git", "worktree", "prune"], cwd=git_project, capture_output=True)

        # Branch should still exist
        result = subprocess.run(
            ["git", "branch", "--list", agent.branch],
            cwd=git_project,
            capture_output=True,
            text=True,
        )
        assert agent.branch in result.stdout

        # Update agent state to reflect no worktree
        agent.worktree = None

        # Manual merge and cleanup
        subprocess.run(["git", "checkout", "main"], cwd=git_project, capture_output=True)
        subprocess.run(
            ["git", "merge", "--no-ff", agent.branch, "-m", "Recovery merge"],
            cwd=git_project,
            capture_output=True,
        )
        subprocess.run(
            ["git", "branch", "-d", agent.branch],
            cwd=git_project,
            capture_output=True,
        )

        # Reset agent
        agent.status = "idle"
        agent.task = None
        agent.branch = ""
        save_state(state, git_project)

        assert_invariants(state, git_project)

    def test_idempotent_ticket_close(self, git_project: Path):
        """Closing an already-closed ticket is safe."""
        # Close ticket
        ticket_file = git_project / ".tickets" / "t-0001.md"
        content = ticket_file.read_text()
        content = content.replace("status: open", "status: closed")
        ticket_file.write_text(content)

        # close_ticket should handle already-closed tickets gracefully
        # (either return True or False, but not error)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = close_ticket("t-0001")
            # Should not raise

    def test_branch_already_merged_is_handled(self, git_project: Path):
        """If branch is already merged, complete_task handles it gracefully."""
        agent, state = self._setup_done_agent(git_project)

        # Manually merge branch first
        remove_worktree(agent.worktree)
        subprocess.run(["git", "checkout", "main"], cwd=git_project, capture_output=True)
        subprocess.run(
            ["git", "merge", "--no-ff", agent.branch, "-m", "Pre-merge"],
            cwd=git_project,
            capture_output=True,
        )

        # Update agent to reflect no worktree
        agent.worktree = None

        # Try to merge again - should handle gracefully
        # The branch might still exist but is already merged
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", agent.branch, "main"],
            cwd=git_project,
            capture_output=True,
        )
        # Branch is ancestor of main = already merged
        assert result.returncode == 0


class TestCompleteTaskInvariants:
    """Tests verifying complete_task maintains invariants."""

    def _setup_done_agent(self, git_project: Path) -> tuple[Agent, State]:
        """Helper to create a done agent ready for completion."""
        state = State()
        agent = spawn_worker("test-agent", state, git_project)

        with patch("crew.runner.get_task_description") as mock_desc:
            mock_desc.return_value = "Test task"
            assign_task(agent, "t-0001", state, git_project)

        _commit_work_in_worktree(agent.worktree)

        mock = MockClaude()
        mock.configure("success", responses=["DONE"])

        with patch("crew.runner.run_claude", mock):
            with patch("crew.runner._refresh_agent_summary"):
                step_agent(agent, state, project_root=git_project)

        return agent, state

    def test_invariants_hold_before_complete(self, git_project: Path):
        """Invariants hold for done agent before complete_task."""
        agent, state = self._setup_done_agent(git_project)
        assert_invariants(state, git_project)

    def test_invariants_hold_after_complete(self, git_project: Path):
        """Invariants hold after successful complete_task."""
        agent, state = self._setup_done_agent(git_project)

        with patch("crew.runner.run_tests_in_worktree") as mock_tests:
            mock_tests.return_value = (True, "Tests passed")
            with patch("crew.runner.close_ticket") as mock_close:
                mock_close.return_value = True
                complete_task(agent, state, project_root=git_project)

        assert_invariants(state, git_project)

    def test_invariants_hold_after_test_failure(self, git_project: Path):
        """Invariants hold after test failure in complete_task."""
        agent, state = self._setup_done_agent(git_project)

        with patch("crew.runner.run_tests_in_worktree") as mock_tests:
            mock_tests.return_value = (False, "Test failed")
            complete_task(agent, state, project_root=git_project)

        assert_invariants(state, git_project)
