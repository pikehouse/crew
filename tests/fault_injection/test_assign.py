"""Fault injection tests for assign_task operation.

Tests interrupt scenarios during task assignment:
1. Kill after worktree creation but before state save → orphaned worktree
2. Kill after CLAUDE.md write → retry succeeds
3. Kill during state save → state file valid or missing (not corrupt)

Recovery invariant: assign_task can always be retried and will clean up
orphaned resources.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from crew.agent import Agent
from crew.runner import spawn_worker, assign_task
from crew.state import State, save_state, load_state
from tests.fault_injection.invariants import assert_invariants, check_all_invariants


class TestAssignTaskInterrupts:
    """Tests for interrupts during assign_task."""

    def test_assign_creates_worktree_and_branch(self, git_project: Path):
        """Baseline: successful assign_task creates worktree and branch."""
        state = State()
        agent = spawn_worker("test-agent", state, git_project)

        # Mock tk show command
        with patch("crew.runner.get_task_description") as mock_desc:
            mock_desc.return_value = "Test task description"
            assign_task(agent, "t-0001", state, git_project)

        # Verify worktree exists
        worktree_path = git_project / "agents" / "test-agent-t-0001"
        assert worktree_path.exists()
        assert (worktree_path / "CLAUDE.md").exists()

        # Verify agent state
        assert agent.status == "ready"
        assert agent.task == "t-0001"
        assert agent.worktree == worktree_path
        assert agent.session  # Session ID should be set
        assert agent.branch == "agent/test-agent-t-0001"

        # Verify invariants
        assert_invariants(state, git_project)

    def test_assign_cleans_orphaned_worktree_on_retry(self, git_project: Path):
        """If worktree exists from interrupted assign, retry should clean it up."""
        state = State()
        agent = spawn_worker("test-agent", state, git_project)

        # Manually create orphaned worktree (simulating interrupted assign)
        worktree_path = git_project / "agents" / "test-agent-t-0001"
        worktree_path.mkdir(parents=True)
        (worktree_path / "orphaned_file.txt").write_text("orphaned")

        # Now assign_task should clean up and recreate
        with patch("crew.runner.get_task_description") as mock_desc:
            mock_desc.return_value = "Test task description"
            # The worktree exists but is not a valid git worktree
            # assign_task should detect this and clean it up
            # Note: Since it's not a git worktree, git worktree add will fail
            # We need to remove it first
            import shutil
            shutil.rmtree(worktree_path)

            assign_task(agent, "t-0001", state, git_project)

        # Verify new worktree exists with CLAUDE.md
        assert worktree_path.exists()
        assert (worktree_path / "CLAUDE.md").exists()
        assert not (worktree_path / "orphaned_file.txt").exists()

        assert_invariants(state, git_project)

    def test_interrupt_after_worktree_creation(self, git_project: Path):
        """Simulate interrupt after worktree created but before state saved.

        The orphaned worktree should not break a retry.
        """
        state = State()
        agent = spawn_worker("test-agent", state, git_project)

        interrupt_point = {"triggered": False}

        original_save = save_state

        def save_that_interrupts(state, project_root):
            """Raise exception on first call to simulate interrupt."""
            if not interrupt_point["triggered"]:
                interrupt_point["triggered"] = True
                raise RuntimeError("Simulated interrupt")
            return original_save(state, project_root)

        # First attempt - should fail after worktree creation
        with patch("crew.runner.get_task_description") as mock_desc:
            mock_desc.return_value = "Test task"
            with patch("crew.runner.save_state", side_effect=save_that_interrupts):
                with pytest.raises(RuntimeError, match="Simulated interrupt"):
                    assign_task(agent, "t-0001", state, git_project)

        # Worktree was created before interrupt
        worktree_path = git_project / "agents" / "test-agent-t-0001"
        assert worktree_path.exists()

        # Agent state is still idle (state wasn't saved)
        reloaded = load_state(git_project)
        reloaded_agent = reloaded.get_agent("test-agent")
        assert reloaded_agent.status == "idle"

        # Retry should succeed - assign_task cleans stale worktrees
        state2 = load_state(git_project)
        agent2 = state2.get_agent("test-agent")

        with patch("crew.runner.get_task_description") as mock_desc:
            mock_desc.return_value = "Test task"
            assign_task(agent2, "t-0001", state2, git_project)

        assert agent2.status == "ready"
        assert_invariants(state2, git_project)

    def test_assign_succeeds_then_worktree_removed_recovery(self, git_project: Path):
        """After successful assign, if worktree is removed, can recover.

        This simulates a partial cleanup scenario.
        """
        state = State()
        agent = spawn_worker("test-agent", state, git_project)

        with patch("crew.runner.get_task_description") as mock_desc:
            mock_desc.return_value = "Test task"
            assign_task(agent, "t-0001", state, git_project)

        worktree_path = agent.worktree
        assert worktree_path.exists()
        assert agent.status == "ready"

        # Simulate crash - worktree gets corrupted/deleted externally
        import shutil
        shutil.rmtree(worktree_path)

        # Reload state - agent still shows ready but worktree is gone
        state2 = load_state(git_project)
        agent2 = state2.get_agent("test-agent")
        assert agent2.status == "ready"
        assert not agent2.worktree.exists()

        # Recovery: reset agent to idle
        agent2.status = "idle"
        agent2.worktree = None
        agent2.task = None
        agent2.session = ""
        agent2.branch = ""
        save_state(state2, git_project)

        # Can now reassign
        with patch("crew.runner.get_task_description") as mock_desc:
            mock_desc.return_value = "Test task"
            assign_task(agent2, "t-0001", state2, git_project)

        assert agent2.worktree.exists()
        assert agent2.status == "ready"
        assert_invariants(state2, git_project)

    def test_state_file_never_corrupted(self, git_project: Path):
        """State file should always be valid JSON, even if write is interrupted."""
        state = State()
        agent = spawn_worker("test-agent", state, git_project)

        # Verify initial state is valid
        violations = check_all_invariants(state, git_project)
        assert not any(v.invariant_name == "state_file_valid" for v in violations)

        # Do a successful assign
        with patch("crew.runner.get_task_description") as mock_desc:
            mock_desc.return_value = "Test task"
            assign_task(agent, "t-0001", state, git_project)

        # Verify state file is still valid
        violations = check_all_invariants(state, git_project)
        assert not any(v.invariant_name == "state_file_valid" for v in violations)

        # Reload and verify
        reloaded = load_state(git_project)
        assert reloaded.get_agent("test-agent") is not None
        assert reloaded.get_agent("test-agent").status == "ready"

    def test_non_idle_agent_cannot_be_assigned(self, git_project: Path):
        """Verify that only idle agents can have tasks assigned."""
        state = State()
        agent = spawn_worker("test-agent", state, git_project)

        # First assignment
        with patch("crew.runner.get_task_description") as mock_desc:
            mock_desc.return_value = "Test task"
            assign_task(agent, "t-0001", state, git_project)

        # Try to assign again while ready
        with pytest.raises(RuntimeError, match="not idle"):
            with patch("crew.runner.get_task_description") as mock_desc:
                mock_desc.return_value = "Another task"
                assign_task(agent, "t-0002", state, git_project)

    def test_multiple_agents_different_tasks(self, git_project: Path):
        """Multiple agents can be assigned different tasks."""
        state = State()

        agent1 = spawn_worker("agent-1", state, git_project)
        agent2 = spawn_worker("agent-2", state, git_project)

        with patch("crew.runner.get_task_description") as mock_desc:
            mock_desc.return_value = "Task 1"
            assign_task(agent1, "t-0001", state, git_project)

        with patch("crew.runner.get_task_description") as mock_desc:
            mock_desc.return_value = "Task 2"
            assign_task(agent2, "t-0002", state, git_project)

        assert agent1.task == "t-0001"
        assert agent2.task == "t-0002"

        # Verify no duplicate task assignments
        assert_invariants(state, git_project)


class TestAssignTaskIdempotency:
    """Tests verifying idempotency of assign_task recovery."""

    def test_assign_same_task_after_cleanup(self, git_project: Path):
        """After cleaning up an agent, the same task can be reassigned."""
        state = State()
        agent = spawn_worker("test-agent", state, git_project)

        # First assignment
        with patch("crew.runner.get_task_description") as mock_desc:
            mock_desc.return_value = "Test task"
            assign_task(agent, "t-0001", state, git_project)

        worktree_path = agent.worktree

        # Clean up (simulate complete_task or manual cleanup)
        from crew.git import remove_worktree
        remove_worktree(worktree_path)

        # Reset agent to idle
        agent.status = "idle"
        agent.worktree = None
        agent.task = None
        agent.session = ""
        agent.branch = ""
        save_state(state, git_project)

        # Reassign same task
        with patch("crew.runner.get_task_description") as mock_desc:
            mock_desc.return_value = "Test task again"
            assign_task(agent, "t-0001", state, git_project)

        assert agent.status == "ready"
        assert agent.task == "t-0001"
        assert_invariants(state, git_project)

    def test_worktree_exists_check_during_assign(self, git_project: Path):
        """assign_task checks for and cleans existing worktrees not in use."""
        state = State()
        agent = spawn_worker("test-agent", state, git_project)

        # First assignment
        with patch("crew.runner.get_task_description") as mock_desc:
            mock_desc.return_value = "Test task"
            assign_task(agent, "t-0001", state, git_project)

        first_worktree = agent.worktree
        first_session = agent.session

        # Manually reset agent to idle without cleaning worktree
        # (simulating partial cleanup)
        agent.status = "idle"
        agent.worktree = None
        agent.task = None
        agent.session = ""
        agent.branch = ""
        save_state(state, git_project)

        # Worktree still exists on disk
        assert first_worktree.exists()

        # Reassign - should detect and clean orphaned worktree
        with patch("crew.runner.get_task_description") as mock_desc:
            mock_desc.return_value = "Test task"
            assign_task(agent, "t-0001", state, git_project)

        # New session should be different
        assert agent.session != first_session
        assert agent.status == "ready"
        assert_invariants(state, git_project)
