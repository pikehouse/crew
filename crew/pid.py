"""PID file management for single-instance enforcement."""

from __future__ import annotations

import os
import signal
from pathlib import Path

PID_FILE = "crew.pid"


def get_pid_file_path(project_root: Path) -> Path:
    """Get the path to the PID file."""
    return project_root / ".crew" / PID_FILE


def is_process_running(pid: int) -> bool:
    """Check if a process with the given PID is running.

    Returns True if the process exists, False otherwise.
    """
    try:
        # Send signal 0 to check if process exists (doesn't actually send a signal)
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        # Process doesn't exist
        return False
    except PermissionError:
        # Process exists but we don't have permission to signal it
        return True


def read_pid_file(project_root: Path) -> int | None:
    """Read the PID from the PID file.

    Returns the PID if file exists and is valid, None otherwise.
    """
    pid_path = get_pid_file_path(project_root)
    if not pid_path.exists():
        return None

    try:
        content = pid_path.read_text().strip()
        return int(content)
    except (ValueError, OSError):
        return None


def write_pid_file(project_root: Path) -> None:
    """Write current process PID to the PID file."""
    pid_path = get_pid_file_path(project_root)
    pid_path.write_text(str(os.getpid()))


def remove_pid_file(project_root: Path) -> None:
    """Remove the PID file if it exists."""
    pid_path = get_pid_file_path(project_root)
    try:
        pid_path.unlink()
    except FileNotFoundError:
        pass


def check_and_acquire_lock(project_root: Path) -> bool:
    """Check for existing crew instance and acquire lock if available.

    Returns:
        True if lock was acquired (we are the primary instance)
        False if another instance is running (we should enter read-only mode)
    """
    existing_pid = read_pid_file(project_root)

    if existing_pid is not None:
        if existing_pid == os.getpid():
            # This is our own PID (shouldn't happen in normal use)
            return True

        if is_process_running(existing_pid):
            # Another crew instance is running
            return False

        # Stale PID file - process no longer exists
        # Clean it up and acquire the lock
        remove_pid_file(project_root)

    # Acquire the lock
    write_pid_file(project_root)
    return True
