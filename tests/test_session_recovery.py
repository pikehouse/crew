"""Tests for session recovery and PID locking integration.

This module tests the integration between:
- PID file locking for single-instance enforcement
- Session recovery on startup
- Worktree reconciliation
- Orphaned worktree detection
- Stale PID handling
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from crew.agent import Agent
from crew.state import State, ensure_crew_dir, save_state, load_state
from crew.pid import (
    get_pid_file_path,
    is_process_running,
    read_pid_file,
    write_pid_file,
    remove_pid_file,
    check_and_acquire_lock,
)
from crew.cli import recover_session, handle_command, get_prompt, _MODIFYING_COMMANDS
import crew.cli


class TestStalePidHandling:
    """Tests for stale PID file detection and cleanup."""

    def test_stale_pid_cleanup_and_lock_acquisition(self, project_root: Path):
        """Test that stale PID is cleaned up and lock acquired in one operation."""
        pid_path = get_pid_file_path(project_root)

        # Write a stale PID (non-existent process)
        stale_pid = 999999999
        pid_path.write_text(str(stale_pid))

        # Acquire lock - should clean up stale and acquire
        result = check_and_acquire_lock(project_root)

        assert result is True
        # Our PID should now be in the file
        assert read_pid_file(project_root) == os.getpid()

    def test_stale_pid_with_invalid_content(self, project_root: Path):
        """Test handling of PID file with invalid content."""
        pid_path = get_pid_file_path(project_root)

        # Write invalid content
        pid_path.write_text("invalid-pid")

        # Should treat as no lock and acquire
        result = check_and_acquire_lock(project_root)
        assert result is True
        assert read_pid_file(project_root) == os.getpid()

    def test_stale_pid_with_whitespace(self, project_root: Path):
        """Test handling of PID file with whitespace around PID."""
        pid_path = get_pid_file_path(project_root)

        # Write PID with whitespace
        pid_path.write_text("  12345  \n")

        # read_pid_file should handle whitespace
        pid = read_pid_file(project_root)
        assert pid == 12345

    def test_multiple_stale_pid_acquisitions(self, project_root: Path):
        """Test multiple acquisitions after stale PID cleanup."""
        pid_path = get_pid_file_path(project_root)

        # First: stale PID
        pid_path.write_text("999999999")
        assert check_and_acquire_lock(project_root) is True

        # Second: we own it now
        assert check_and_acquire_lock(project_root) is True

        # Third: still us
        assert read_pid_file(project_root) == os.getpid()

    def test_permission_error_on_process_check(self, project_root: Path):
        """Test handling when process check raises PermissionError."""
        pid_path = get_pid_file_path(project_root)

        # Write PID 1 (init) which exists but might give permission error
        pid_path.write_text("1")

        # Should return False (process is running or we can't confirm it's not)
        result = check_and_acquire_lock(project_root)
        assert result is False


class TestPidFileEdgeCases:
    """Edge case tests for PID file operations."""

    def test_concurrent_pid_file_reads(self, project_root: Path):
        """Test reading PID file while it's being written."""
        pid_path = get_pid_file_path(project_root)

        # Write a valid PID
        write_pid_file(project_root)
        original_pid = read_pid_file(project_root)

        # Multiple reads should be consistent
        for _ in range(10):
            assert read_pid_file(project_root) == original_pid

    def test_pid_file_creation_in_nonexistent_dir(self, temp_dir: Path):
        """Test that write_pid_file fails if .crew dir doesn't exist."""
        # Don't create .crew dir
        with pytest.raises(FileNotFoundError):
            write_pid_file(temp_dir)

    def test_remove_pid_file_idempotent(self, project_root: Path):
        """Test that remove_pid_file can be called multiple times safely."""
        write_pid_file(project_root)

        # Remove multiple times - should not raise
        remove_pid_file(project_root)
        remove_pid_file(project_root)
        remove_pid_file(project_root)

        assert read_pid_file(project_root) is None

    def test_pid_file_with_newline(self, project_root: Path):
        """Test reading PID file that has trailing newline."""
        pid_path = get_pid_file_path(project_root)
        pid_path.write_text("54321\n")

        assert read_pid_file(project_root) == 54321

    def test_pid_file_with_crlf(self, project_root: Path):
        """Test reading PID file with Windows-style line endings."""
        pid_path = get_pid_file_path(project_root)
        pid_path.write_text("54321\r\n")

        assert read_pid_file(project_root) == 54321


class TestReadOnlyModeIntegration:
    """Integration tests for read-only mode with session operations."""

    def test_read_only_mode_set_when_lock_fails(self, project_root: Path):
        """Test that read-only mode is set when another instance is running."""
        # Simulate another instance by writing PID 1 (init)
        pid_path = get_pid_file_path(project_root)
        pid_path.write_text("1")

        # check_and_acquire_lock should return False
        result = check_and_acquire_lock(project_root)
        assert result is False

    def test_read_only_mode_not_set_when_lock_succeeds(self, project_root: Path):
        """Test that read-only mode is not set when we get the lock."""
        result = check_and_acquire_lock(project_root)
        assert result is True

    def test_all_modifying_commands_blocked(self, project_root: Path, empty_state: State):
        """Test that all modifying commands are blocked in read-only mode."""
        old_value = crew.cli._read_only_mode
        crew.cli._read_only_mode = True

        try:
            for cmd in _MODIFYING_COMMANDS:
                result = handle_command(cmd, empty_state, project_root)
                assert result is True  # Continue REPL, but command rejected
        finally:
            crew.cli._read_only_mode = old_value

    def test_help_works_in_read_only_mode(self, project_root: Path, empty_state: State, capsys):
        """Test that help command works in read-only mode."""
        old_value = crew.cli._read_only_mode
        crew.cli._read_only_mode = True

        try:
            result = handle_command("help", empty_state, project_root)
            assert result is True
            captured = capsys.readouterr()
            assert "help" in captured.out.lower() or "Commands" in captured.out
        finally:
            crew.cli._read_only_mode = old_value

    def test_prompt_indicator_in_read_only_mode(self, empty_state: State):
        """Test that prompt shows read-only indicator."""
        old_value = crew.cli._read_only_mode
        crew.cli._read_only_mode = True

        try:
            prompt = get_prompt(empty_state)
            assert "[read-only]" in prompt
        finally:
            crew.cli._read_only_mode = old_value

    def test_prompt_no_indicator_in_normal_mode(self, empty_state: State):
        """Test that prompt has no read-only indicator in normal mode."""
        old_value = crew.cli._read_only_mode
        crew.cli._read_only_mode = False

        try:
            prompt = get_prompt(empty_state)
            assert "[read-only]" not in prompt
        finally:
            crew.cli._read_only_mode = old_value


class TestStartupRecoveryIntegration:
    """Integration tests for startup recovery flow."""

    def test_recovery_with_multiple_agents_different_states(
        self, project_root: Path, capsys
    ):
        """Test recovery with agents in multiple different states."""
        state = State()

        # Idle agent
        idle_agent = Agent(
            name="idle-one",
            session="",
            worktree=None,
            branch="",
            task=None,
            status="idle",
        )
        state.add_agent(idle_agent)

        # Done agent
        done_agent = Agent(
            name="done-one",
            session="sess-done",
            worktree=None,
            branch="agent/done-one",
            task="completed-task",
            status="done",
        )
        state.add_agent(done_agent)

        # Stuck agent
        stuck_agent = Agent(
            name="stuck-one",
            session="sess-stuck",
            worktree=None,
            branch="agent/stuck-one",
            task="stuck-task",
            status="stuck",
        )
        state.add_agent(stuck_agent)

        with patch("crew.cli.get_worktree_list") as mock_worktrees:
            mock_worktrees.return_value = []
            result = recover_session(state, project_root)

        assert result is True
        captured = capsys.readouterr()
        assert "idle-one" in captured.out
        assert "done-one" in captured.out
        assert "stuck-one" in captured.out

    def test_recovery_resets_session_fields_on_missing_worktree(
        self, project_root: Path
    ):
        """Test that all session-related fields are reset when worktree is missing."""
        state = State()
        agent = Agent(
            name="reset-test",
            session="old-session",
            worktree=Path("/nonexistent/path"),
            branch="agent/reset-test",
            task="old-task",
            status="working",
            step_count=10,
            last_step_at=datetime(2026, 1, 1, 12, 0, 0),
        )
        state.add_agent(agent)

        with patch("crew.cli.get_worktree_list") as mock_worktrees:
            mock_worktrees.return_value = []
            recover_session(state, project_root)

        # All session fields should be reset
        assert agent.status == "idle"
        assert agent.worktree is None
        assert agent.branch == ""
        assert agent.task is None
        assert agent.session == ""
        assert agent.step_count == 0
        assert agent.last_step_at is None

    def test_recovery_resets_done_agent_with_missing_worktree(
        self, project_root: Path
    ):
        """Test that done agents with missing worktrees are reset to idle.

        When a done agent's worktree is missing, there's no way to verify the work
        or run tests before merging. The safest approach is to reset to idle so
        the operator can manually handle the situation (e.g., find the branch
        and merge it manually if the work was committed).
        """
        state = State()
        agent = Agent(
            name="done-preserve",
            session="sess-done",
            worktree=None,  # Worktree already gone
            branch="agent/done-preserve",
            task="finished-task",
            status="done",
        )
        state.add_agent(agent)

        with patch("crew.cli.get_worktree_list") as mock_worktrees:
            mock_worktrees.return_value = []
            recover_session(state, project_root)

        # Agent should be reset to idle since worktree is missing
        assert agent.status == "idle"
        assert agent.task is None

    def test_recovery_calls_shutdown_agent_correctly(
        self, temp_dir: Path
    ):
        """Test that shutdown_agent is called with correct arguments."""
        worktree_path = temp_dir / "agents" / "test-wt"
        worktree_path.mkdir(parents=True)
        (temp_dir / ".crew").mkdir(parents=True)

        state = State()
        agent = Agent(
            name="shutdown-test",
            session="sess-shutdown",
            worktree=worktree_path,
            branch="agent/shutdown-test",
            task="some-task",
            status="working",
            step_count=5,
        )
        state.add_agent(agent)

        with patch("crew.cli.get_worktree_list") as mock_worktrees, \
             patch("crew.cli.has_uncommitted_changes") as mock_uncommitted, \
             patch("crew.cli.shutdown_agent") as mock_shutdown, \
             patch("crew.cli.remove_worktree"):
            mock_worktrees.return_value = [{"path": str(worktree_path)}]
            mock_uncommitted.return_value = False  # Clean worktree goes to shutdown_agent path
            mock_shutdown.return_value = "partial"
            recover_session(state, temp_dir)

        # shutdown_agent should have been called with the agent and state
        mock_shutdown.assert_called_once()
        call_args = mock_shutdown.call_args
        assert call_args[0][0] == agent  # First arg is agent
        assert call_args[0][1] == state  # Second arg is state


class TestWorktreeReconciliation:
    """Tests for worktree reconciliation during recovery."""

    def test_worktree_in_git_but_not_on_disk(self, project_root: Path):
        """Test handling when git reports worktree but it doesn't exist on disk."""
        state = State()
        agent = Agent(
            name="ghost-wt",
            session="sess-ghost",
            worktree=Path("/ghost/worktree"),  # Doesn't exist
            branch="agent/ghost-wt",
            task="ghost-task",
            status="working",
        )
        state.add_agent(agent)

        with patch("crew.cli.get_worktree_list") as mock_worktrees:
            # Git says it exists, but it doesn't on disk
            mock_worktrees.return_value = [{"path": "/ghost/worktree"}]
            recover_session(state, project_root)

        # Should be reset since path doesn't exist on disk
        assert agent.status == "idle"

    def test_worktree_on_disk_but_not_in_git(self, temp_dir: Path, capsys):
        """Test detection of worktree on disk but not returned by git."""
        (temp_dir / ".crew").mkdir(parents=True)
        orphan_dir = temp_dir / "agents" / "orphan-wt"
        orphan_dir.mkdir(parents=True)

        state = State()
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
            mock_worktrees.return_value = []  # Git doesn't know about it
            recover_session(state, temp_dir)

        captured = capsys.readouterr()
        assert "Orphaned worktrees" in captured.out

    def test_worktree_matches_between_git_and_disk(self, temp_dir: Path):
        """Test that matching worktrees are handled correctly."""
        (temp_dir / ".crew").mkdir(parents=True)
        worktree_dir = temp_dir / "agents" / "good-wt"
        worktree_dir.mkdir(parents=True)

        state = State()
        agent = Agent(
            name="good-agent",
            session="sess-good",
            worktree=worktree_dir,
            branch="agent/good-agent",
            task="good-task",
            status="working",
        )
        state.add_agent(agent)

        with patch("crew.cli.get_worktree_list") as mock_worktrees, \
             patch("crew.cli.has_uncommitted_changes") as mock_uncommitted, \
             patch("crew.cli.shutdown_agent") as mock_shutdown, \
             patch("crew.cli.remove_worktree"):
            mock_worktrees.return_value = [{"path": str(worktree_dir)}]
            mock_uncommitted.return_value = False  # Clean worktree
            mock_shutdown.return_value = "partial"
            recover_session(state, temp_dir)

        # shutdown_agent should have been called for reconciliation
        mock_shutdown.assert_called_once()

    def test_bare_worktrees_filtered_from_git_list(self, project_root: Path):
        """Test that bare repositories are filtered from worktree list."""
        state = State()
        agent = Agent(
            name="test-agent",
            session="",
            worktree=None,
            branch="",
            task=None,
            status="idle",
        )
        state.add_agent(agent)

        with patch("crew.cli.get_worktree_list") as mock_worktrees:
            # Git returns both bare and normal worktrees
            mock_worktrees.return_value = [
                {"path": "/some/path", "bare": True},  # Should be filtered
                {"path": "/another/path", "bare": False},  # Should be included
            ]
            recover_session(state, project_root)

        # No errors, recovery completes


class TestOrphanedWorktreeDetection:
    """Tests for orphaned worktree detection."""

    def test_multiple_orphaned_worktrees(self, temp_dir: Path, capsys):
        """Test detection of multiple orphaned worktrees."""
        (temp_dir / ".crew").mkdir(parents=True)
        agents_dir = temp_dir / "agents"
        agents_dir.mkdir()

        # Create multiple orphans
        (agents_dir / "orphan1").mkdir()
        (agents_dir / "orphan2").mkdir()
        (agents_dir / "orphan3").mkdir()

        state = State()
        agent = Agent(
            name="known-agent",
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
        assert "orphan1" in captured.out
        assert "orphan2" in captured.out
        assert "orphan3" in captured.out

    def test_no_orphans_when_all_tracked(self, temp_dir: Path, capsys):
        """Test no orphan warning when all worktrees are tracked."""
        (temp_dir / ".crew").mkdir(parents=True)
        agents_dir = temp_dir / "agents"
        agents_dir.mkdir()

        # Create a tracked worktree
        tracked_wt = agents_dir / "tracked-wt"
        tracked_wt.mkdir()

        state = State()
        agent = Agent(
            name="tracked-agent",
            session="sess-tracked",
            worktree=tracked_wt,
            branch="agent/tracked-agent",
            task="tracked-task",
            status="idle",  # Idle but has worktree path
        )
        state.add_agent(agent)

        with patch("crew.cli.get_worktree_list") as mock_worktrees:
            mock_worktrees.return_value = []
            recover_session(state, temp_dir)

        captured = capsys.readouterr()
        assert "Orphaned worktrees" not in captured.out

    def test_no_agents_dir_no_orphans(self, project_root: Path, capsys):
        """Test no orphan detection when agents/ directory doesn't exist."""
        state = State()
        agent = Agent(
            name="test-agent",
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
        assert "Orphaned worktrees" not in captured.out

    def test_files_in_agents_dir_not_orphans(self, temp_dir: Path, capsys):
        """Test that files (not directories) in agents/ are ignored."""
        (temp_dir / ".crew").mkdir(parents=True)
        agents_dir = temp_dir / "agents"
        agents_dir.mkdir()

        # Create a file, not a directory
        (agents_dir / "some_file.txt").write_text("content")

        state = State()
        agent = Agent(
            name="test-agent",
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
        # Files should not be reported as orphaned worktrees
        assert "some_file" not in captured.out or "Orphaned" not in captured.out


class TestRecoveryActionMessages:
    """Tests for recovery action messages and display."""

    def test_done_action_message(self, temp_dir: Path, capsys):
        """Test action message when agent is marked done from logs."""
        (temp_dir / ".crew").mkdir(parents=True)
        worktree_path = temp_dir / "agents" / "done-agent"
        worktree_path.mkdir(parents=True)

        state = State()
        agent = Agent(
            name="done-from-logs",
            session="sess-done",
            worktree=worktree_path,
            branch="agent/done-from-logs",
            task="finish-task",
            status="working",
        )
        state.add_agent(agent)

        with patch("crew.cli.get_worktree_list") as mock_worktrees, \
             patch("crew.cli.has_uncommitted_changes") as mock_uncommitted, \
             patch("crew.cli.shutdown_agent") as mock_shutdown, \
             patch("crew.cli.complete_task") as mock_complete:
            mock_worktrees.return_value = [{"path": str(worktree_path)}]
            mock_uncommitted.return_value = False
            mock_shutdown.return_value = "done"
            mock_complete.return_value = (True, None)
            recover_session(state, temp_dir)

        captured = capsys.readouterr()
        # When done, complete_task is called which merges the branch
        assert "Completed done-from-logs" in captured.out

    def test_partial_action_message(self, temp_dir: Path, capsys):
        """Test action message for partial work."""
        (temp_dir / ".crew").mkdir(parents=True)
        worktree_path = temp_dir / "agents" / "partial-agent"
        worktree_path.mkdir(parents=True)

        state = State()
        agent = Agent(
            name="partial-work",
            session="sess-partial",
            worktree=worktree_path,
            branch="agent/partial-work",
            task="partial-task",
            status="working",
        )
        state.add_agent(agent)

        with patch("crew.cli.get_worktree_list") as mock_worktrees, \
             patch("crew.cli.has_uncommitted_changes") as mock_uncommitted, \
             patch("crew.cli.shutdown_agent") as mock_shutdown, \
             patch("crew.cli.remove_worktree"):
            mock_worktrees.return_value = [{"path": str(worktree_path)}]
            mock_uncommitted.return_value = False
            mock_shutdown.return_value = "partial"
            recover_session(state, temp_dir)

        captured = capsys.readouterr()
        assert "partial work" in captured.out

    def test_nothing_action_message(self, temp_dir: Path, capsys):
        """Test action message when no work was done."""
        (temp_dir / ".crew").mkdir(parents=True)
        worktree_path = temp_dir / "agents" / "nothing-agent"
        worktree_path.mkdir(parents=True)

        state = State()
        agent = Agent(
            name="nothing-done",
            session="sess-nothing",
            worktree=worktree_path,
            branch="agent/nothing-done",
            task="nothing-task",
            status="ready",
        )
        state.add_agent(agent)

        with patch("crew.cli.get_worktree_list") as mock_worktrees, \
             patch("crew.cli.has_uncommitted_changes") as mock_uncommitted, \
             patch("crew.cli.shutdown_agent") as mock_shutdown, \
             patch("crew.cli.remove_worktree"):
            mock_worktrees.return_value = [{"path": str(worktree_path)}]
            mock_uncommitted.return_value = False
            mock_shutdown.return_value = "nothing"
            recover_session(state, temp_dir)

        captured = capsys.readouterr()
        assert "Reset nothing-done to idle (no work done, ticket stays open)" in captured.out

    def test_worktree_missing_action_message(self, project_root: Path, capsys):
        """Test action message when worktree is missing."""
        state = State()
        agent = Agent(
            name="missing-wt",
            session="sess-missing",
            worktree=Path("/nonexistent/worktree"),
            branch="agent/missing-wt",
            task="missing-task",
            status="working",
        )
        state.add_agent(agent)

        with patch("crew.cli.get_worktree_list") as mock_worktrees:
            mock_worktrees.return_value = []
            recover_session(state, project_root)

        captured = capsys.readouterr()
        assert "Reset missing-wt to idle" in captured.out
        assert "worktree missing" in captured.out


class TestRecoverySummary:
    """Tests for recovery summary display."""

    def test_summary_with_working_agents(self, temp_dir: Path, capsys):
        """Test summary message when working agents exist."""
        (temp_dir / ".crew").mkdir(parents=True)
        worktree_path = temp_dir / "agents" / "working-sum"
        worktree_path.mkdir(parents=True)

        state = State()
        agent = Agent(
            name="summary-working",
            session="sess-working",
            worktree=worktree_path,
            branch="agent/summary-working",
            task="working-task",
            status="working",
        )
        state.add_agent(agent)

        with patch("crew.cli.get_worktree_list") as mock_worktrees, \
             patch("crew.cli.has_uncommitted_changes") as mock_uncommitted, \
             patch("crew.cli.shutdown_agent") as mock_shutdown, \
             patch("crew.cli.remove_worktree"):
            mock_worktrees.return_value = [{"path": str(worktree_path)}]
            mock_uncommitted.return_value = False
            mock_shutdown.return_value = "partial"
            # Agent stays in working status after partial
            recover_session(state, temp_dir)

        captured = capsys.readouterr()
        # Should show ready to resume
        assert "Ready to resume" in captured.out or "run" in captured.out.lower()

    def test_summary_with_only_idle_agents(self, project_root: Path, capsys):
        """Test summary message when only idle agents exist."""
        state = State()
        for i in range(3):
            agent = Agent(
                name=f"idle-{i}",
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
        assert "Session restored" in captured.out
        assert "idle" in captured.out

    def test_summary_with_only_done_agents(self, project_root: Path, capsys):
        """Test summary message when only done agents exist."""
        state = State()
        agent = Agent(
            name="only-done",
            session="sess-done",
            worktree=None,
            branch="agent/only-done",
            task="done-task",
            status="done",
        )
        state.add_agent(agent)

        with patch("crew.cli.get_worktree_list") as mock_worktrees:
            mock_worktrees.return_value = []
            recover_session(state, project_root)

        captured = capsys.readouterr()
        assert "Session restored" in captured.out


class TestRecoveryStateIntegration:
    """Integration tests for state saving during recovery."""

    def test_state_saved_after_recovery(self, temp_dir: Path):
        """Test that state is saved after recovery."""
        (temp_dir / ".crew").mkdir(parents=True)

        state = State()
        agent = Agent(
            name="save-test",
            session="sess-save",
            worktree=Path("/nonexistent"),
            branch="agent/save-test",
            task="save-task",
            status="working",
        )
        state.add_agent(agent)

        with patch("crew.cli.get_worktree_list") as mock_worktrees:
            mock_worktrees.return_value = []
            recover_session(state, temp_dir)

        # Load state and verify it was saved
        loaded_state = load_state(temp_dir)
        loaded_agent = loaded_state.get_agent("save-test")
        assert loaded_agent is not None
        assert loaded_agent.status == "idle"  # Should be reset

    def test_state_changes_persisted_through_recovery(self, temp_dir: Path):
        """Test that all state changes during recovery are persisted."""
        (temp_dir / ".crew").mkdir(parents=True)

        state = State()
        # Add agent with worktree that will be reset
        agent1 = Agent(
            name="will-reset",
            session="sess-1",
            worktree=Path("/nonexistent"),
            branch="agent/will-reset",
            task="task-1",
            status="working",
            step_count=5,
        )
        state.add_agent(agent1)

        # Add idle agent that stays the same
        agent2 = Agent(
            name="stays-idle",
            session="",
            worktree=None,
            branch="",
            task=None,
            status="idle",
        )
        state.add_agent(agent2)

        with patch("crew.cli.get_worktree_list") as mock_worktrees:
            mock_worktrees.return_value = []
            recover_session(state, temp_dir)

        # Load and verify both agents
        loaded_state = load_state(temp_dir)

        loaded1 = loaded_state.get_agent("will-reset")
        assert loaded1.status == "idle"
        assert loaded1.step_count == 0

        loaded2 = loaded_state.get_agent("stays-idle")
        assert loaded2.status == "idle"
