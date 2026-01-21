"""Tests for crew.crew_logging module."""

from __future__ import annotations

from pathlib import Path

import pytest

from crew.crew_logging import (
    clear_agent_logs,
    get_log_dir,
    get_next_log_number,
    get_session_file_path,
    read_latest_log,
    read_log_tail,
    write_log,
)


class TestGetLogDir:
    """Test get_log_dir function."""

    def test_get_log_dir_creates_directory(self, project_root: Path):
        """Test that get_log_dir creates the log directory."""
        log_dir = get_log_dir("test-agent", project_root)

        assert log_dir.exists()
        assert log_dir.is_dir()
        assert log_dir == project_root / ".crew" / "logs" / "test-agent"

    def test_get_log_dir_is_idempotent(self, project_root: Path):
        """Test that get_log_dir can be called multiple times safely."""
        get_log_dir("test-agent", project_root)
        log_dir = get_log_dir("test-agent", project_root)

        assert log_dir.exists()


class TestGetNextLogNumber:
    """Test get_next_log_number function."""

    def test_returns_1_when_no_logs(self, project_root: Path):
        """Test that first log number is 1 when no logs exist."""
        num = get_next_log_number("test-agent", project_root)
        assert num == 1

    def test_increments_after_existing_logs(self, project_root: Path):
        """Test that log number increments correctly."""
        log_dir = get_log_dir("test-agent", project_root)
        (log_dir / "001-init.log").write_text("test")
        (log_dir / "002-step.log").write_text("test")

        num = get_next_log_number("test-agent", project_root)
        assert num == 3


class TestWriteLog:
    """Test write_log function."""

    def test_write_log_creates_file(self, project_root: Path):
        """Test that write_log creates a log file."""
        log_path = write_log(
            agent_name="test-agent",
            log_type="init",
            prompt="test prompt",
            output="test output",
            session="test-session",
            step=1,
            project_root=project_root,
        )

        assert log_path.exists()
        assert log_path.name == "001-init.log"

    def test_write_log_content_format(self, project_root: Path):
        """Test that write_log writes correct format."""
        log_path = write_log(
            agent_name="test-agent",
            log_type="step",
            prompt="my prompt",
            output="my output",
            session="sess-123",
            step=5,
            project_root=project_root,
        )

        content = log_path.read_text()
        assert "=== CREW LOG ===" in content
        assert "Agent: test-agent" in content
        assert "Step: 5" in content
        assert "my prompt" in content
        assert "my output" in content
        assert "Session: sess-123" in content
        assert "=== END ===" in content


class TestReadLatestLog:
    """Test read_latest_log function."""

    def test_returns_none_when_no_logs(self, project_root: Path):
        """Test that read_latest_log returns None when no logs exist."""
        get_log_dir("test-agent", project_root)
        content = read_latest_log("test-agent", project_root)
        assert content is None

    def test_returns_latest_log_content(self, project_root: Path):
        """Test that read_latest_log returns the highest-numbered log."""
        log_dir = get_log_dir("test-agent", project_root)
        (log_dir / "001-init.log").write_text("first log")
        (log_dir / "002-step.log").write_text("second log")
        (log_dir / "003-step.log").write_text("latest log")

        content = read_latest_log("test-agent", project_root)
        assert content == "latest log"


class TestReadLogTail:
    """Test read_log_tail function."""

    def test_returns_none_when_no_logs(self, project_root: Path):
        """Test that read_log_tail returns None when no logs exist."""
        get_log_dir("test-agent", project_root)
        content = read_log_tail("test-agent", project_root=project_root)
        assert content is None

    def test_extracts_output_section(self, project_root: Path):
        """Test that read_log_tail extracts the output section."""
        write_log(
            agent_name="test-agent",
            log_type="step",
            prompt="prompt",
            output="line1\nline2\nline3",
            session="sess",
            step=1,
            project_root=project_root,
        )

        content = read_log_tail("test-agent", lines=10, project_root=project_root)
        assert "line1" in content
        assert "line2" in content
        assert "line3" in content


class TestClearAgentLogs:
    """Test clear_agent_logs function."""

    def test_clears_all_log_files(self, project_root: Path):
        """Test that clear_agent_logs removes all log files."""
        # Create some log files
        log_dir = get_log_dir("test-agent", project_root)
        (log_dir / "001-init.log").write_text("log 1")
        (log_dir / "002-step.log").write_text("log 2")
        (log_dir / "003-step.log").write_text("log 3")

        # Verify logs exist
        assert len(list(log_dir.glob("*.log"))) == 3

        # Clear logs
        clear_agent_logs("test-agent", project_root)

        # Verify logs are cleared
        assert len(list(log_dir.glob("*.log"))) == 0

    def test_clears_only_log_files(self, project_root: Path):
        """Test that clear_agent_logs only removes .log files."""
        log_dir = get_log_dir("test-agent", project_root)
        (log_dir / "001-step.log").write_text("log")
        (log_dir / "other.txt").write_text("other file")

        clear_agent_logs("test-agent", project_root)

        # Log should be deleted, other file should remain
        assert not (log_dir / "001-step.log").exists()
        assert (log_dir / "other.txt").exists()

    def test_does_nothing_when_no_logs(self, project_root: Path):
        """Test that clear_agent_logs doesn't fail when no logs exist."""
        get_log_dir("test-agent", project_root)

        # Should not raise
        clear_agent_logs("test-agent", project_root)

    def test_log_numbers_restart_after_clear(self, project_root: Path):
        """Test that log numbers restart at 1 after clearing."""
        # Create logs
        log_dir = get_log_dir("test-agent", project_root)
        (log_dir / "001-init.log").write_text("log 1")
        (log_dir / "002-step.log").write_text("log 2")

        # Verify next number would be 3
        assert get_next_log_number("test-agent", project_root) == 3

        # Clear logs
        clear_agent_logs("test-agent", project_root)

        # Next number should be 1 again
        assert get_next_log_number("test-agent", project_root) == 1

    def test_read_latest_log_returns_none_after_clear(self, project_root: Path):
        """Test that read_latest_log returns None after clearing."""
        # Create a log
        write_log(
            agent_name="test-agent",
            log_type="step",
            prompt="prompt",
            output="old task output",
            session="old-session",
            step=1,
            project_root=project_root,
        )

        # Verify log exists
        assert read_latest_log("test-agent", project_root) is not None

        # Clear logs
        clear_agent_logs("test-agent", project_root)

        # No logs should exist now
        assert read_latest_log("test-agent", project_root) is None


class TestGetSessionFilePath:
    """Test get_session_file_path function."""

    def test_returns_none_when_file_does_not_exist(self, tmp_path: Path):
        """Test that get_session_file_path returns None when session file doesn't exist."""
        worktree = tmp_path / "fake-worktree"
        worktree.mkdir()

        result = get_session_file_path(worktree, "nonexistent-session")
        assert result is None

    def test_returns_path_when_file_exists(self, tmp_path: Path):
        """Test that get_session_file_path returns path when session file exists."""
        # Create a fake worktree
        worktree = tmp_path / "test-worktree"
        worktree.mkdir()

        # Create the Claude projects directory structure
        worktree_str = str(worktree.resolve())
        project_dir_name = worktree_str.replace("/", "-")
        claude_dir = Path.home() / ".claude" / "projects" / project_dir_name
        claude_dir.mkdir(parents=True, exist_ok=True)

        # Create a session file
        session_id = "test-session-123"
        session_file = claude_dir / f"{session_id}.jsonl"
        session_file.write_text('{"type": "test"}\n')

        try:
            result = get_session_file_path(worktree, session_id)
            assert result is not None
            assert result == session_file
        finally:
            # Cleanup
            session_file.unlink()
            claude_dir.rmdir()

    def test_path_conversion_replaces_slashes(self, tmp_path: Path):
        """Test that worktree path slashes are converted to dashes."""
        worktree = tmp_path / "sub" / "dir" / "worktree"
        worktree.mkdir(parents=True)

        # The function should look for a path with dashes instead of slashes
        # We don't need to create the actual file to test the path conversion
        # Just verify it doesn't crash
        result = get_session_file_path(worktree, "any-session")
        # Should return None since file doesn't exist
        assert result is None
