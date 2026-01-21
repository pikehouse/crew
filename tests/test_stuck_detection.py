"""Tests for stuck_detection module."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from crew.stuck_detection import (
    StuckReason,
    StuckTracker,
    check_if_stuck,
    get_backoff_seconds,
    record_error,
    record_success,
    reset_tracker,
    should_backoff,
    update_output_hash,
)


class TestGetBackoffSeconds:
    """Tests for get_backoff_seconds function."""

    def test_exponential_backoff(self):
        """Backoff increases exponentially."""
        assert get_backoff_seconds(1) == 2
        assert get_backoff_seconds(2) == 4
        assert get_backoff_seconds(3) == 8
        assert get_backoff_seconds(4) == 16
        assert get_backoff_seconds(5) == 32

    def test_max_backoff_60_seconds(self):
        """Backoff is capped at 60 seconds."""
        assert get_backoff_seconds(6) == 60
        assert get_backoff_seconds(10) == 60
        assert get_backoff_seconds(100) == 60


class TestCheckIfStuck:
    """Tests for check_if_stuck function."""

    def test_stuck_by_step_limit(self):
        """Agent is stuck when step count exceeds limit."""
        agent = MagicMock()
        agent.step_count = 20
        agent.status = "working"

        result = check_if_stuck(agent, step_limit=20)

        assert result is not None
        assert result.reason == "step_limit"
        assert "20 steps" in result.details

    def test_not_stuck_if_done(self):
        """Agent is not stuck if status is done, even with high step count."""
        agent = MagicMock()
        agent.step_count = 100
        agent.status = "done"

        result = check_if_stuck(agent, step_limit=20)

        assert result is None

    def test_stuck_by_error_limit(self):
        """Agent is stuck when error count exceeds limit."""
        agent = MagicMock()
        agent.step_count = 5
        agent.status = "working"

        tracker = StuckTracker(error_count=5)

        result = check_if_stuck(agent, tracker=tracker, error_limit=5)

        assert result is not None
        assert result.reason == "error_limit"
        assert "5 consecutive errors" in result.details

    def test_stuck_by_stall(self):
        """Agent is stuck when output hasn't changed."""
        agent = MagicMock()
        agent.step_count = 5
        agent.status = "working"

        # Output unchanged for 6 minutes
        tracker = StuckTracker(
            last_output_change=time.time() - 6 * 60,
            last_output_hash="abc123",
        )

        result = check_if_stuck(agent, tracker=tracker, stall_minutes=5)

        assert result is not None
        assert result.reason == "stalled"
        assert "6 minutes" in result.details

    def test_not_stuck_below_thresholds(self):
        """Agent is not stuck when below all thresholds."""
        agent = MagicMock()
        agent.step_count = 10
        agent.status = "working"

        tracker = StuckTracker(
            error_count=2,
            last_output_change=time.time() - 60,  # 1 minute ago
        )

        result = check_if_stuck(agent, tracker=tracker)

        assert result is None


class TestBackoffTracking:
    """Tests for backoff-related functions."""

    def test_should_backoff_true_during_backoff(self):
        """should_backoff returns True during backoff period."""
        tracker = StuckTracker(backoff_until=time.time() + 10)
        assert should_backoff(tracker) is True

    def test_should_backoff_false_after_backoff(self):
        """should_backoff returns False after backoff expires."""
        tracker = StuckTracker(backoff_until=time.time() - 1)
        assert should_backoff(tracker) is False

    def test_record_error_increments_count(self):
        """record_error increments error count."""
        tracker = StuckTracker()

        record_error(tracker)
        assert tracker.error_count == 1

        record_error(tracker)
        assert tracker.error_count == 2

    def test_record_error_sets_backoff(self):
        """record_error sets appropriate backoff time."""
        tracker = StuckTracker()

        before = time.time()
        record_error(tracker)
        after = time.time()

        # First error: 2 second backoff
        assert tracker.backoff_until >= before + 2
        assert tracker.backoff_until <= after + 2

    def test_record_success_resets_tracking(self):
        """record_success resets error count and backoff."""
        tracker = StuckTracker(error_count=5, backoff_until=time.time() + 100)

        record_success(tracker)

        assert tracker.error_count == 0
        assert tracker.backoff_until == 0.0


class TestOutputHashTracking:
    """Tests for output hash tracking."""

    def test_update_output_hash_detects_change(self):
        """update_output_hash returns True when content changes."""
        tracker = StuckTracker()

        result = update_output_hash(tracker, "content 1")
        assert result is True

        result = update_output_hash(tracker, "content 2")
        assert result is True

    def test_update_output_hash_detects_no_change(self):
        """update_output_hash returns False when content is same."""
        tracker = StuckTracker()

        update_output_hash(tracker, "same content")
        result = update_output_hash(tracker, "same content")

        assert result is False

    def test_update_output_hash_tracks_last_change_time(self):
        """update_output_hash tracks when content last changed."""
        tracker = StuckTracker()

        before = time.time()
        update_output_hash(tracker, "content")
        after = time.time()

        assert tracker.last_output_change >= before
        assert tracker.last_output_change <= after


class TestResetTracker:
    """Tests for reset_tracker function."""

    def test_reset_clears_all_fields(self):
        """reset_tracker clears all tracking fields."""
        tracker = StuckTracker(
            error_count=5,
            backoff_until=time.time() + 100,
            last_output_hash="abc123",
            last_output_change=time.time() - 300,
        )

        reset_tracker(tracker)

        assert tracker.error_count == 0
        assert tracker.backoff_until == 0.0
        assert tracker.last_output_hash is None
        assert tracker.last_output_change is None
