"""Mock Claude for fault injection testing.

Provides a configurable mock that can simulate various failure modes:
- success: Normal response with optional DONE marker
- timeout: Raises TimeoutError
- partial_response: Returns partial output (simulates interrupted response)
- session_conflict: Returns session ID error in stderr
- rate_limit: Raises RuntimeError for rate limiting
- hang: Blocks until interrupted (for testing kill scenarios)

Unlike fake_claude.py (which is a CLI replacement for e2e tests), this
module provides Python classes for unit tests with fine-grained control.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class MockResponse:
    """Represents a mock Claude response."""

    result: str = ""
    input_tokens: int = 100
    output_tokens: int = 150
    cost_usd: float = 0.001
    stderr: str = ""

    def to_dict(self) -> dict:
        """Convert to the dict format that run_claude returns."""
        return {
            "result": self.result,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "stderr": self.stderr,
        }


class MockClaude:
    """Configurable mock for Claude CLI with failure mode support.

    Usage:
        mock = MockClaude()

        # Simple success
        mock.configure("success", responses=["Working on it...", "DONE"])

        # Fail after N calls
        mock.configure("timeout", fail_after_n=2)

        # Custom per-call behavior
        mock.configure("custom", custom_fn=my_handler)

    Then patch run_claude with mock:
        with patch('crew.runner.run_claude', mock):
            step_agent(...)
    """

    # Available failure modes
    FAILURE_MODES = [
        "success",          # Normal response
        "timeout",          # Raises TimeoutError
        "partial_response", # Returns partial/truncated output
        "session_conflict", # Session ID error in stderr
        "rate_limit",       # Raises RuntimeError for rate limit
        "hang",             # Blocks until interrupted
        "custom",           # Uses custom_fn callback
    ]

    def __init__(self):
        """Initialize with default success mode."""
        self._mode = "success"
        self._fail_after_n = 0
        self._call_count = 0
        self._responses: list[str] = ["Working on task..."]
        self._custom_fn: Callable | None = None
        self._hang_event: threading.Event | None = None
        self._calls: list[dict] = []  # Record all calls for inspection
        self._default_tokens = (100, 150)
        self._default_cost = 0.001

    def configure(
        self,
        mode: str,
        fail_after_n: int = 0,
        responses: list[str] | None = None,
        custom_fn: Callable | None = None,
    ) -> "MockClaude":
        """Configure the mock behavior.

        Args:
            mode: One of FAILURE_MODES
            fail_after_n: Number of successful calls before failure (0 = fail immediately)
            responses: List of response strings (cycled through)
            custom_fn: Custom handler for "custom" mode (receives call args)

        Returns:
            self for chaining
        """
        if mode not in self.FAILURE_MODES:
            raise ValueError(f"Unknown mode: {mode}. Must be one of {self.FAILURE_MODES}")

        self._mode = mode
        self._fail_after_n = fail_after_n
        self._call_count = 0
        self._calls = []

        if responses is not None:
            self._responses = responses
        if custom_fn is not None:
            self._custom_fn = custom_fn

        # Create hang event for hang mode
        if mode == "hang":
            self._hang_event = threading.Event()

        return self

    def reset(self) -> "MockClaude":
        """Reset call count and recorded calls."""
        self._call_count = 0
        self._calls = []
        return self

    def interrupt_hang(self) -> None:
        """Interrupt a hanging call (for testing kill scenarios)."""
        if self._hang_event:
            self._hang_event.set()

    @property
    def calls(self) -> list[dict]:
        """Get list of all calls made to this mock."""
        return self._calls.copy()

    @property
    def call_count(self) -> int:
        """Get number of calls made."""
        return self._call_count

    def __call__(
        self,
        prompt: str,
        cwd,
        session: str | None = None,
        is_new_session: bool = False,
        timeout: int = 300,
        model: str | None = None,
    ) -> dict:
        """Handle a call to run_claude.

        Args match the signature of runner.run_claude().
        """
        # Record the call
        call_info = {
            "prompt": prompt,
            "cwd": str(cwd),
            "session": session,
            "is_new_session": is_new_session,
            "timeout": timeout,
            "model": model,
            "call_number": self._call_count,
        }
        self._calls.append(call_info)

        # Check if we should succeed or fail
        should_fail = (
            self._mode != "success" and
            self._call_count >= self._fail_after_n
        )

        self._call_count += 1

        if should_fail:
            return self._handle_failure(call_info)
        else:
            return self._handle_success(call_info)

    def _handle_success(self, call_info: dict) -> dict:
        """Handle a successful call."""
        # Get response text (cycle through responses list)
        response_idx = (call_info["call_number"]) % len(self._responses)
        result_text = self._responses[response_idx]

        return MockResponse(
            result=result_text,
            input_tokens=self._default_tokens[0],
            output_tokens=self._default_tokens[1],
            cost_usd=self._default_cost,
        ).to_dict()

    def _handle_failure(self, call_info: dict) -> dict:
        """Handle a failure based on configured mode."""
        if self._mode == "timeout":
            raise TimeoutError(f"Claude command timed out after {call_info['timeout']}s")

        elif self._mode == "partial_response":
            # Return truncated response
            return MockResponse(
                result="Starting to work on...",  # Truncated
                input_tokens=50,
                output_tokens=20,
                cost_usd=0.0003,
            ).to_dict()

        elif self._mode == "session_conflict":
            # Return success but with session error in stderr
            return MockResponse(
                result="",
                stderr="Error: Session ID already in use",
            ).to_dict()

        elif self._mode == "rate_limit":
            raise RuntimeError("Rate limit exceeded. Please try again later.")

        elif self._mode == "hang":
            # Block until interrupted
            if self._hang_event:
                self._hang_event.wait()
            raise TimeoutError("Hang interrupted")

        elif self._mode == "custom":
            if self._custom_fn:
                return self._custom_fn(call_info)
            raise RuntimeError("Custom mode requires custom_fn")

        else:
            # Shouldn't reach here
            return self._handle_success(call_info)


@dataclass
class MockClaudeBuilder:
    """Fluent builder for creating MockClaude instances.

    Example:
        mock = (MockClaudeBuilder()
            .succeeds_times(2)
            .then_times_out()
            .with_responses(["Step 1...", "Step 2..."])
            .build())
    """

    _steps: list[tuple[str, int]] = field(default_factory=list)
    _responses: list[str] = field(default_factory=lambda: ["Working..."])

    def succeeds_times(self, n: int) -> "MockClaudeBuilder":
        """Add N successful calls."""
        self._steps.append(("success", n))
        return self

    def then_times_out(self) -> "MockClaudeBuilder":
        """Add a timeout failure."""
        self._steps.append(("timeout", 1))
        return self

    def then_rate_limits(self) -> "MockClaudeBuilder":
        """Add a rate limit failure."""
        self._steps.append(("rate_limit", 1))
        return self

    def then_session_conflicts(self) -> "MockClaudeBuilder":
        """Add a session conflict failure."""
        self._steps.append(("session_conflict", 1))
        return self

    def with_responses(self, responses: list[str]) -> "MockClaudeBuilder":
        """Set response texts."""
        self._responses = responses
        return self

    def build(self) -> MockClaude:
        """Build the MockClaude with custom call sequence.

        Creates a mock that uses the custom mode to handle the
        configured sequence of successes and failures.
        """
        steps = self._steps.copy()
        responses = self._responses.copy()

        # Build sequence of outcomes
        outcomes: list[tuple[str, str]] = []
        for mode, count in steps:
            for _ in range(count):
                outcomes.append((mode, responses[len(outcomes) % len(responses)]))

        def custom_handler(call_info: dict) -> dict:
            idx = call_info["call_number"]
            if idx >= len(outcomes):
                # Past configured sequence - default to success
                return MockResponse(
                    result=responses[idx % len(responses)]
                ).to_dict()

            mode, response_text = outcomes[idx]

            if mode == "success":
                return MockResponse(result=response_text).to_dict()
            elif mode == "timeout":
                raise TimeoutError("Claude command timed out")
            elif mode == "rate_limit":
                raise RuntimeError("Rate limit exceeded")
            elif mode == "session_conflict":
                return MockResponse(
                    result="",
                    stderr="Error: Session ID already in use",
                ).to_dict()
            else:
                return MockResponse(result=response_text).to_dict()

        mock = MockClaude()
        mock.configure("custom", custom_fn=custom_handler)
        mock._responses = responses
        return mock


# Convenience functions for common patterns
def mock_always_succeeds(responses: list[str] | None = None) -> MockClaude:
    """Create a mock that always succeeds with given responses."""
    mock = MockClaude()
    if responses:
        mock.configure("success", responses=responses)
    return mock


def mock_times_out_after(n: int, responses: list[str] | None = None) -> MockClaude:
    """Create a mock that times out after N successful calls."""
    mock = MockClaude()
    mock.configure("timeout", fail_after_n=n, responses=responses or ["Working..."])
    return mock


def mock_session_conflict_after(n: int) -> MockClaude:
    """Create a mock that has session conflict after N calls."""
    mock = MockClaude()
    mock.configure("session_conflict", fail_after_n=n)
    return mock


def mock_done_on_step(step: int) -> MockClaude:
    """Create a mock that returns DONE on the specified step (0-indexed)."""
    responses = ["Working on task..."] * step + ["DONE\n\nTask completed."]
    return mock_always_succeeds(responses)
