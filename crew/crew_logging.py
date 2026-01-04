"""Logging for crew agent interactions."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from crew.state import ensure_crew_dir


def get_log_dir(agent_name: str, project_root: Path | None = None) -> Path:
    """Get the log directory for an agent."""
    crew_dir = ensure_crew_dir(project_root)
    log_dir = crew_dir / "logs" / agent_name
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def clear_agent_logs(agent_name: str, project_root: Path | None = None) -> None:
    """Clear all log files for an agent.

    This should be called when a new task is assigned to ensure the dashboard
    doesn't show stale logs from previous tasks.

    Args:
        agent_name: Name of the agent
        project_root: Optional project root path
    """
    log_dir = get_log_dir(agent_name, project_root)
    for log_file in log_dir.glob("*.log"):
        log_file.unlink()


def get_next_log_number(agent_name: str, project_root: Path | None = None) -> int:
    """Get the next log file number for an agent."""
    log_dir = get_log_dir(agent_name, project_root)
    existing = list(log_dir.glob("*.log"))
    if not existing:
        return 1
    # Extract numbers from filenames like "001-init.log", "002-step.log"
    numbers = []
    for f in existing:
        try:
            num = int(f.stem.split("-")[0])
            numbers.append(num)
        except (ValueError, IndexError):
            pass
    return max(numbers, default=0) + 1


def write_log(
    agent_name: str,
    log_type: str,
    prompt: str,
    output: str,
    session: str,
    step: int,
    project_root: Path | None = None,
) -> Path:
    """Write a log entry for an agent interaction.

    Args:
        agent_name: Name of the agent
        log_type: Type of log (e.g., "init", "step")
        prompt: The prompt sent to Claude
        output: The full output from Claude
        session: The session ID
        step: The step number
        project_root: Optional project root path

    Returns:
        Path to the log file
    """
    log_dir = get_log_dir(agent_name, project_root)
    log_num = get_next_log_number(agent_name, project_root)
    log_file = log_dir / f"{log_num:03d}-{log_type}.log"

    timestamp = datetime.now().isoformat()

    content = f"""=== CREW LOG ===
Agent: {agent_name}
Time: {timestamp}
Step: {step}
Prompt: {prompt}
Session: {session}
---
{output}
=== END ===
"""

    log_file.write_text(content)
    return log_file


def read_latest_log(agent_name: str, project_root: Path | None = None) -> str | None:
    """Read the latest log file for an agent."""
    log_dir = get_log_dir(agent_name, project_root)
    logs = sorted(log_dir.glob("*.log"))
    if not logs:
        return None
    return logs[-1].read_text()


def read_log_tail(agent_name: str, lines: int = 30, project_root: Path | None = None) -> str | None:
    """Read the last N lines of the latest log for an agent."""
    content = read_latest_log(agent_name, project_root)
    if not content:
        return None

    # Extract the output section (between --- and === END ===)
    parts = content.split("---\n", 1)
    if len(parts) < 2:
        return content

    output = parts[1].rsplit("=== END ===", 1)[0]
    log_lines = output.strip().split("\n")
    return "\n".join(log_lines[-lines:])


def read_all_logs(agent_name: str, project_root: Path | None = None) -> str | None:
    """Read all log files for an agent, concatenated in order.

    Returns:
        Combined content of all log files, or None if no logs exist.
    """
    log_dir = get_log_dir(agent_name, project_root)
    logs = sorted(log_dir.glob("*.log"))
    if not logs:
        return None

    contents = []
    for log_file in logs:
        contents.append(log_file.read_text())
    return "\n".join(contents)
