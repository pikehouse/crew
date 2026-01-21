"""Consolidated stuck detection for crew agents.

This module provides a single source of truth for detecting stuck agents,
replacing the three separate mechanisms that were spread across runner.py
and cli.py:
1. Step count limit (was in runner.py)
2. Consecutive error limit with backoff (was in cli.py)
3. Output stall detection (was in cli.py)
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from crew.agent import Agent


@dataclass
class StuckReason:
    """Describes why an agent is considered stuck."""

    reason: Literal["step_limit", "error_limit", "stalled"]
    details: str


@dataclass
class StuckTracker:
    """Tracks stuck detection state for a single agent.

    This consolidates the tracking that was previously spread across
    multiple dicts (_error_counts, _error_backoff_until, _output_hashes).
    """

    error_count: int = 0
    backoff_until: float = 0.0
    last_output_hash: str | None = None
    last_output_change: float | None = None


def get_backoff_seconds(error_count: int) -> int:
    """Return backoff seconds for error count.

    Implements exponential backoff: 2s, 4s, 8s, 16s, 32s, 60s max.

    Args:
        error_count: Number of consecutive errors

    Returns:
        Seconds to wait before next retry
    """
    return min(60, 2 ** error_count)


def check_if_stuck(
    agent: "Agent",
    tracker: StuckTracker | None = None,
    step_limit: int = 20,
    error_limit: int = 5,
    stall_minutes: int = 5,
) -> StuckReason | None:
    """Single function to check all stuck conditions.

    Args:
        agent: The agent to check
        tracker: Optional StuckTracker with error/stall history
        step_limit: Max steps before marking stuck (default 20)
        error_limit: Max consecutive errors before stuck (default 5)
        stall_minutes: Minutes of no output change before stuck (default 5)

    Returns:
        StuckReason if stuck, None otherwise
    """
    # Check step count limit
    if agent.step_count >= step_limit and agent.status != "done":
        return StuckReason(
            reason="step_limit",
            details=f"Reached {agent.step_count} steps (limit: {step_limit})",
        )

    # Check error count if tracker provided
    if tracker and tracker.error_count >= error_limit:
        return StuckReason(
            reason="error_limit",
            details=f"{tracker.error_count} consecutive errors (limit: {error_limit})",
        )

    # Check output stall if tracker provided
    if tracker and tracker.last_output_change is not None:
        now = time.time()
        stalled_seconds = now - tracker.last_output_change
        if stalled_seconds > stall_minutes * 60:
            return StuckReason(
                reason="stalled",
                details=f"No output change for {int(stalled_seconds / 60)} minutes",
            )

    return None


def should_backoff(tracker: StuckTracker) -> bool:
    """Check if agent should be skipped due to error backoff.

    Args:
        tracker: StuckTracker for the agent

    Returns:
        True if current time is before backoff_until
    """
    return time.time() < tracker.backoff_until


def record_error(tracker: StuckTracker) -> int:
    """Record an error and update backoff timing.

    Args:
        tracker: StuckTracker for the agent

    Returns:
        New error count
    """
    tracker.error_count += 1
    backoff_secs = get_backoff_seconds(tracker.error_count)
    tracker.backoff_until = time.time() + backoff_secs
    return tracker.error_count


def record_success(tracker: StuckTracker) -> None:
    """Record a successful step, resetting error tracking.

    Args:
        tracker: StuckTracker for the agent
    """
    tracker.error_count = 0
    tracker.backoff_until = 0.0


def update_output_hash(tracker: StuckTracker, content: str) -> bool:
    """Update output hash tracking and check for stall.

    Args:
        tracker: StuckTracker for the agent
        content: The content to hash (typically last N lines of output)

    Returns:
        True if content changed, False if unchanged
    """
    current_hash = hashlib.md5(content.encode()).hexdigest()
    now = time.time()

    if current_hash != tracker.last_output_hash:
        # Output changed
        tracker.last_output_hash = current_hash
        tracker.last_output_change = now
        return True
    else:
        # Output unchanged - if no previous change time, set it now
        if tracker.last_output_change is None:
            tracker.last_output_change = now
        return False


def reset_tracker(tracker: StuckTracker) -> None:
    """Reset all tracking for an agent.

    Call this when an agent's state is reset (e.g., after recovery).

    Args:
        tracker: StuckTracker to reset
    """
    tracker.error_count = 0
    tracker.backoff_until = 0.0
    tracker.last_output_hash = None
    tracker.last_output_change = None
