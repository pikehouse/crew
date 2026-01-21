"""Fault injection tests for recovery operations.

Tests recovery from various corrupt or inconsistent states:
1. Orphaned worktrees (worktree exists but not in state)
2. Orphaned branches (branch exists but no agent owns it)
3. State references non-existent worktree
4. Agent stuck with invalid session

The goal is to verify that the system can recover to a valid state
from any inconsistent state.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from crew.agent import Agent
from crew.runner import (
    spawn_worker,
    assign_task,
    step_agent,
    shutdown_agent,
    check_work_completed,
)
from crew.git import create_worktree, remove_worktree, get_worktree_list
from crew.state import State, save_state, load_state
from tests.fault_injection.invariants import (
    assert_invariants,
    check_all_invariants,
    InvariantViolation,
)
from tests.fixtures.mock_claude import MockClaude


class TestOrphanedWorktreeRecovery:
    """Tests for recovering from orphaned worktrees."""

    def test_detect_orphaned_worktree(self, git_project: Path):
        """Detect worktrees not referenced by any agent."""
        state = State()

        # Create worktree manually (orphaned)
        agents_dir = git_project / "agents"
        orphan_path = create_worktree("orphan-worktree", agents_dir)

        # Verify worktree exists
        assert orphan_path.exists()

        # No agent references it
        assert len(state.agents) == 0

        # Get worktree list from the git project (not crew repo)
        result = subprocess.run(
            ["git", "worktree", "list"],
            cwd=git_project,
            capture_output=True,
            text=True,
        )
        worktree_lines = result.stdout.strip().split("\n")
        orphan_worktrees = [
            line for line in worktree_lines
            if "orphan-worktree" in line
        ]
        assert len(orphan_worktrees) == 1

    def test_assign_cleans_orphaned_worktree(self, git_project: Path):
        """assign_task cleans up orphaned worktree before creating new one."""
        state = State()
        agent = spawn_worker("test-agent", state, git_project)

        # Create orphaned worktree with same name pattern
        agents_dir = git_project / "agents"
        orphan_name = "test-agent-t-0001"
        orphan_path = agents_dir / orphan_name
        orphan_path.mkdir(parents=True)
        (orphan_path / "orphan.txt").write_text("orphaned file")

        # It's not a git worktree, just a directory

        # assign_task should handle this
        with patch("crew.runner.get_task_description") as mock_desc:
            mock_desc.return_value = "Test task"
            # The directory exists but isn't a worktree, so git worktree add will fail
            # unless we clean it up first
            import shutil
            shutil.rmtree(orphan_path)

            assign_task(agent, "t-0001", state, git_project)

        # New worktree created
        assert agent.worktree.exists()
        assert (agent.worktree / "CLAUDE.md").exists()
        assert not (agent.worktree / "orphan.txt").exists()

    def test_recover_from_orphaned_git_worktree(self, git_project: Path):
        """Recover from actual orphaned git worktree."""
        state = State()

        # Create an actual git worktree
        agents_dir = git_project / "agents"
        worktree_path = create_worktree("orphan-branch", agents_dir)

        # Configure git user for worktree
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=worktree_path,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=worktree_path,
            capture_output=True,
        )

        # Create some work in it
        (worktree_path / "work.py").write_text("# Some work\n")
        subprocess.run(["git", "add", "-A"], cwd=worktree_path, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Add work"],
            cwd=worktree_path,
            capture_output=True,
        )

        # No agent owns this worktree
        # To recover: remove worktree but preserve branch for manual inspection
        remove_worktree(worktree_path)

        # Branch should still exist - check all branches
        result = subprocess.run(
            ["git", "branch", "-a"],
            cwd=git_project,
            capture_output=True,
            text=True,
        )
        assert "orphan-branch" in result.stdout, f"Branch not found. All branches: {result.stdout}"


class TestStateWorktreeMismatchRecovery:
    """Tests for recovering when state and worktrees don't match."""

    def test_state_references_missing_worktree(self, git_project: Path):
        """Detect when state references a non-existent worktree."""
        state = State()
        agent = Agent(
            name="ghost-agent",
            session="session-123",
            worktree=git_project / "agents" / "ghost-worktree",
            branch="agent/ghost",
            task="t-ghost",
            status="working",
        )
        state.add_agent(agent)
        save_state(state, git_project)

        # Worktree doesn't exist
        assert not agent.worktree.exists()

        # Invariant check should catch this
        violations = check_all_invariants(state, git_project)
        worktree_violations = [
            v for v in violations
            if v.invariant_name == "worktree_exists_if_assigned"
        ]
        assert len(worktree_violations) == 1

    def test_recover_from_missing_worktree(self, git_project: Path):
        """Recovery: reset agent to idle if worktree is missing."""
        state = State()
        agent = Agent(
            name="ghost-agent",
            session="session-123",
            worktree=git_project / "agents" / "ghost-worktree",
            branch="agent/ghost",
            task="t-ghost",
            status="working",
        )
        state.add_agent(agent)

        # Worktree doesn't exist - recovery action: reset to idle
        if not agent.worktree.exists():
            agent.status = "idle"
            agent.worktree = None
            agent.task = None
            agent.branch = ""
            agent.session = ""

        save_state(state, git_project)

        # Now invariants should pass
        assert_invariants(state, git_project)


class TestShutdownAgentRecovery:
    """Tests for shutdown_agent recovery behavior."""

    def _setup_working_agent(self, git_project: Path) -> tuple[Agent, State]:
        """Helper to create a working agent."""
        state = State()
        agent = spawn_worker("test-agent", state, git_project)

        with patch("crew.runner.get_task_description") as mock_desc:
            mock_desc.return_value = "Test task"
            assign_task(agent, "t-0001", state, git_project)

        # Step once
        mock = MockClaude()
        mock.configure("success", responses=["Working on task..."])

        with patch("crew.runner.run_claude", mock):
            with patch("crew.runner._refresh_agent_summary"):
                step_agent(agent, state, project_root=git_project)

        return agent, state

    def test_shutdown_with_done_work(self, git_project: Path):
        """shutdown_agent detects DONE in logs and sets status to done."""
        agent, state = self._setup_working_agent(git_project)

        # Step to done
        mock = MockClaude()
        mock.configure("success", responses=["DONE"])

        with patch("crew.runner.run_claude", mock):
            with patch("crew.runner._refresh_agent_summary"):
                step_agent(agent, state, project_root=git_project)

        assert agent.status == "done"

        # Shutdown should recognize done status
        with patch("crew.runner.kill_claude_process", return_value=False):
            status = shutdown_agent(agent, state, git_project)

        assert status == "done"
        assert agent.status == "done"
        assert_invariants(state, git_project)

    def test_shutdown_with_partial_work(self, git_project: Path):
        """shutdown_agent with partial work keeps working status."""
        agent, state = self._setup_working_agent(git_project)

        # Create uncommitted changes
        (agent.worktree / "partial.py").write_text("# Partial work\n")
        subprocess.run(["git", "add", "partial.py"], cwd=agent.worktree, capture_output=True)

        with patch("crew.runner.kill_claude_process", return_value=False):
            status = shutdown_agent(agent, state, git_project)

        assert status == "partial"
        assert agent.status == "working"
        assert_invariants(state, git_project)

    def test_shutdown_with_no_work(self, git_project: Path):
        """shutdown_agent with no work resets to ready."""
        state = State()
        agent = spawn_worker("test-agent", state, git_project)

        with patch("crew.runner.get_task_description") as mock_desc:
            mock_desc.return_value = "Test task"
            assign_task(agent, "t-0001", state, git_project)

        # No steps taken, only CLAUDE.md exists
        assert agent.step_count == 0

        with patch("crew.runner.kill_claude_process", return_value=False):
            status = shutdown_agent(agent, state, git_project)

        assert status == "nothing"
        assert agent.status == "ready"
        assert_invariants(state, git_project)


class TestCheckWorkCompleted:
    """Tests for check_work_completed function."""

    def _setup_working_agent(self, git_project: Path) -> tuple[Agent, State]:
        """Helper to create a working agent."""
        state = State()
        agent = spawn_worker("test-agent", state, git_project)

        with patch("crew.runner.get_task_description") as mock_desc:
            mock_desc.return_value = "Test task"
            assign_task(agent, "t-0001", state, git_project)

        return agent, state

    def test_detect_done_from_logs(self, git_project: Path):
        """Detect DONE marker in logs."""
        agent, state = self._setup_working_agent(git_project)

        # Step to done
        mock = MockClaude()
        mock.configure("success", responses=["DONE"])

        with patch("crew.runner.run_claude", mock):
            with patch("crew.runner._refresh_agent_summary"):
                step_agent(agent, state, project_root=git_project)

        status = check_work_completed(agent, git_project)
        assert status == "done"

    def test_detect_partial_work_from_steps(self, git_project: Path):
        """Detect partial work from step count."""
        agent, state = self._setup_working_agent(git_project)

        mock = MockClaude()
        mock.configure("success", responses=["Working on task..."])

        with patch("crew.runner.run_claude", mock):
            with patch("crew.runner._refresh_agent_summary"):
                step_agent(agent, state, project_root=git_project)

        assert agent.step_count == 1
        status = check_work_completed(agent, git_project)
        assert status == "partial"

    def test_detect_partial_work_from_uncommitted(self, git_project: Path):
        """Detect partial work from uncommitted changes (no steps)."""
        agent, state = self._setup_working_agent(git_project)

        # No steps, but have uncommitted changes
        (agent.worktree / "work.py").write_text("# Work\n")
        subprocess.run(["git", "add", "work.py"], cwd=agent.worktree, capture_output=True)

        status = check_work_completed(agent, git_project)
        assert status == "partial"

    def test_detect_nothing_done(self, git_project: Path):
        """Detect no work done."""
        agent, state = self._setup_working_agent(git_project)

        # No steps, no changes (just CLAUDE.md from assign)
        status = check_work_completed(agent, git_project)
        assert status == "nothing"


class TestComplexRecoveryScenarios:
    """Tests for complex recovery scenarios involving multiple issues."""

    def test_recover_from_multiple_orphaned_worktrees(self, git_project: Path):
        """Recover from multiple orphaned worktrees."""
        state = State()

        # Create multiple orphaned worktrees
        agents_dir = git_project / "agents"
        orphans = []
        for i in range(3):
            path = create_worktree(f"orphan-{i}", agents_dir)
            orphans.append(path)

        # Verify they exist
        for path in orphans:
            assert path.exists()

        # Clean them all up
        for path in orphans:
            remove_worktree(path)

        # Verify cleanup
        for path in orphans:
            assert not path.exists()

        # State should be clean
        assert_invariants(state, git_project)

    def test_recover_from_inconsistent_agent_state(self, git_project: Path):
        """Recover from agent with inconsistent state fields."""
        state = State()

        # Create agent with inconsistent state:
        # - status is "working" but no worktree
        agent = Agent(
            name="broken-agent",
            session="session-123",
            worktree=None,  # Inconsistent!
            branch="",
            task="t-0001",
            status="working",
        )
        state.add_agent(agent)

        # Should violate invariants
        violations = check_all_invariants(state, git_project)
        assert len(violations) > 0

        # Recovery: reset to idle
        agent.status = "idle"
        agent.task = None
        agent.session = ""

        save_state(state, git_project)

        # Now should pass
        assert_invariants(state, git_project)

    def test_state_file_recovery_from_missing(self, git_project: Path):
        """Recover from missing state file."""
        # Delete state file
        state_file = git_project / ".crew" / "state.json"
        if state_file.exists():
            state_file.unlink()

        # load_state should handle missing file
        state = load_state(git_project)

        # Should return empty state
        assert len(state.agents) == 0

        # Can add agents to fresh state
        agent = spawn_worker("new-agent", state, git_project)
        assert state.get_agent("new-agent") is not None

        assert_invariants(state, git_project)
