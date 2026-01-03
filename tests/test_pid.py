"""Tests for crew.pid module - PID file locking for single-instance enforcement."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from crew.pid import (
    get_pid_file_path,
    is_process_running,
    read_pid_file,
    write_pid_file,
    remove_pid_file,
    check_and_acquire_lock,
    PID_FILE,
)


class TestGetPidFilePath:
    """Test the get_pid_file_path function."""

    def test_returns_correct_path(self, project_root: Path):
        """Test that PID file path is in .crew directory."""
        pid_path = get_pid_file_path(project_root)
        assert pid_path == project_root / ".crew" / PID_FILE

    def test_pid_file_name_is_crew_pid(self):
        """Test that the PID file name is crew.pid."""
        assert PID_FILE == "crew.pid"


class TestIsProcessRunning:
    """Test the is_process_running function."""

    def test_current_process_is_running(self):
        """Test that current process is detected as running."""
        assert is_process_running(os.getpid()) is True

    def test_nonexistent_process_is_not_running(self):
        """Test that nonexistent PID is detected as not running."""
        # Use a very high PID that's unlikely to exist
        assert is_process_running(999999999) is False

    def test_pid_one_is_running(self):
        """Test that PID 1 (init) is detected as running."""
        # PID 1 should always exist on Unix systems
        assert is_process_running(1) is True


class TestReadPidFile:
    """Test the read_pid_file function."""

    def test_returns_none_when_no_file(self, project_root: Path):
        """Test that None is returned when PID file doesn't exist."""
        assert read_pid_file(project_root) is None

    def test_returns_pid_when_file_exists(self, project_root: Path):
        """Test that PID is returned when file exists."""
        pid_path = get_pid_file_path(project_root)
        pid_path.write_text("12345")
        assert read_pid_file(project_root) == 12345

    def test_returns_none_for_invalid_content(self, project_root: Path):
        """Test that None is returned for non-numeric content."""
        pid_path = get_pid_file_path(project_root)
        pid_path.write_text("not-a-number")
        assert read_pid_file(project_root) is None

    def test_returns_none_for_empty_file(self, project_root: Path):
        """Test that None is returned for empty file."""
        pid_path = get_pid_file_path(project_root)
        pid_path.write_text("")
        assert read_pid_file(project_root) is None


class TestWritePidFile:
    """Test the write_pid_file function."""

    def test_writes_current_pid(self, project_root: Path):
        """Test that current process PID is written."""
        write_pid_file(project_root)
        pid_path = get_pid_file_path(project_root)
        assert pid_path.exists()
        assert pid_path.read_text().strip() == str(os.getpid())

    def test_overwrites_existing_file(self, project_root: Path):
        """Test that existing PID file is overwritten."""
        pid_path = get_pid_file_path(project_root)
        pid_path.write_text("99999")
        write_pid_file(project_root)
        assert pid_path.read_text().strip() == str(os.getpid())


class TestRemovePidFile:
    """Test the remove_pid_file function."""

    def test_removes_existing_file(self, project_root: Path):
        """Test that existing PID file is removed."""
        pid_path = get_pid_file_path(project_root)
        pid_path.write_text("12345")
        assert pid_path.exists()
        remove_pid_file(project_root)
        assert not pid_path.exists()

    def test_handles_missing_file_gracefully(self, project_root: Path):
        """Test that missing PID file doesn't cause error."""
        pid_path = get_pid_file_path(project_root)
        assert not pid_path.exists()
        # Should not raise
        remove_pid_file(project_root)
        assert not pid_path.exists()


class TestCheckAndAcquireLock:
    """Test the check_and_acquire_lock function."""

    def test_acquires_lock_when_no_pid_file(self, project_root: Path):
        """Test that lock is acquired when no PID file exists."""
        result = check_and_acquire_lock(project_root)
        assert result is True
        # PID file should now exist with current PID
        assert read_pid_file(project_root) == os.getpid()

    def test_returns_false_when_another_process_running(self, project_root: Path):
        """Test that lock is not acquired when another process owns it."""
        # Write PID 1 (init) which is always running
        pid_path = get_pid_file_path(project_root)
        pid_path.write_text("1")

        result = check_and_acquire_lock(project_root)
        assert result is False
        # PID file should still contain the original PID
        assert read_pid_file(project_root) == 1

    def test_acquires_lock_when_stale_pid(self, project_root: Path):
        """Test that lock is acquired when PID file contains stale PID."""
        # Write a very high PID that doesn't exist
        pid_path = get_pid_file_path(project_root)
        pid_path.write_text("999999999")

        result = check_and_acquire_lock(project_root)
        assert result is True
        # PID file should now contain our PID
        assert read_pid_file(project_root) == os.getpid()

    def test_returns_true_when_own_pid(self, project_root: Path):
        """Test that lock is acquired when we already own it."""
        # Write our own PID
        pid_path = get_pid_file_path(project_root)
        pid_path.write_text(str(os.getpid()))

        result = check_and_acquire_lock(project_root)
        assert result is True


class TestPidLockingIntegration:
    """Integration tests for PID locking with CLI."""

    def test_read_only_mode_blocks_modifying_commands(self, project_root: Path, empty_state):
        """Test that modifying commands are blocked in read-only mode."""
        from crew.cli import handle_command, _MODIFYING_COMMANDS
        import crew.cli

        # Set read-only mode
        old_value = crew.cli._read_only_mode
        crew.cli._read_only_mode = True

        try:
            for cmd in _MODIFYING_COMMANDS:
                result = handle_command(cmd, empty_state, project_root)
                # Should return True (continue REPL) but command should be rejected
                assert result is True
        finally:
            crew.cli._read_only_mode = old_value

    def test_read_only_mode_allows_read_commands(self, project_root: Path, empty_state):
        """Test that read-only commands still work in read-only mode."""
        from crew.cli import handle_command
        import crew.cli

        # Set read-only mode
        old_value = crew.cli._read_only_mode
        crew.cli._read_only_mode = True

        try:
            # These should work
            assert handle_command("help", empty_state, project_root) is True
            assert handle_command("dashboard", empty_state, project_root) is True
            assert handle_command("d", empty_state, project_root) is True
            assert handle_command("ps", empty_state, project_root) is True
            # Quit should still work
            assert handle_command("quit", empty_state, project_root) is False
        finally:
            crew.cli._read_only_mode = old_value

    def test_prompt_shows_read_only_indicator(self, empty_state):
        """Test that prompt shows [read-only] when in read-only mode."""
        from crew.cli import get_prompt
        import crew.cli

        # Set read-only mode
        old_value = crew.cli._read_only_mode
        crew.cli._read_only_mode = True

        try:
            prompt = get_prompt(empty_state)
            assert "[read-only]" in prompt
            assert "crew>" in prompt
        finally:
            crew.cli._read_only_mode = old_value

    def test_prompt_no_read_only_indicator_normally(self, empty_state):
        """Test that prompt doesn't show [read-only] in normal mode."""
        from crew.cli import get_prompt
        import crew.cli

        # Ensure not in read-only mode
        old_value = crew.cli._read_only_mode
        crew.cli._read_only_mode = False

        try:
            prompt = get_prompt(empty_state)
            assert "[read-only]" not in prompt
            assert "crew>" in prompt
        finally:
            crew.cli._read_only_mode = old_value
