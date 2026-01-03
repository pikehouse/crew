#!/usr/bin/env python3
"""Fake Claude CLI for e2e testing.

Mimics the behavior of `claude --print --output-format json`.
Reads canned responses from tests/fixtures/responses/*.json files.
Supports multi-step workflows that end in DONE.

Usage:
    fake_claude.py --print --output-format json -p "prompt"
    fake_claude.py --print --output-format json --session-id <id> -p "prompt"
    fake_claude.py --print --output-format json --resume <id> -p "prompt"

Response files are named by session ID and step number:
    responses/{session_id}_step{n}.json

Or use a default response file:
    responses/default.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


# Directory where response files are stored
FIXTURES_DIR = Path(__file__).parent
RESPONSES_DIR = FIXTURES_DIR / "responses"

# Track session state in a temp file
SESSION_STATE_FILE = Path("/tmp/fake_claude_sessions.json")


def load_session_state() -> dict[str, int]:
    """Load the current step count for each session."""
    if SESSION_STATE_FILE.exists():
        try:
            return json.loads(SESSION_STATE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_session_state(state: dict[str, int]) -> None:
    """Save session state to disk."""
    SESSION_STATE_FILE.write_text(json.dumps(state))


def get_next_step(session_id: str) -> int:
    """Get and increment the step counter for a session."""
    state = load_session_state()
    current_step = state.get(session_id, 0)
    state[session_id] = current_step + 1
    save_session_state(state)
    return current_step


def reset_session(session_id: str) -> None:
    """Reset session step counter (for new sessions)."""
    state = load_session_state()
    state[session_id] = 0
    save_session_state(state)


def find_response_file(session_id: str | None, step: int) -> Path | None:
    """Find the appropriate response file for this session/step.

    Search order:
    1. {session_id}_step{step}.json - specific step for this session
    2. step{step}.json - generic step file
    3. default.json - fallback
    """
    if session_id:
        # Try session-specific response
        specific = RESPONSES_DIR / f"{session_id}_step{step}.json"
        if specific.exists():
            return specific

    # Try generic step response
    generic_step = RESPONSES_DIR / f"step{step}.json"
    if generic_step.exists():
        return generic_step

    # Fall back to default
    default = RESPONSES_DIR / "default.json"
    if default.exists():
        return default

    return None


def create_default_response(prompt: str, session_id: str | None, step: int) -> dict[str, Any]:
    """Create a default response with realistic structure."""
    # Generate realistic token counts
    input_tokens = len(prompt.split()) * 3 + 50  # Rough estimate
    output_tokens = 150 + (step * 50)  # Grows slightly with steps

    # Determine if this should be the "DONE" step (after 3 steps by default)
    is_done = step >= 3

    result_text = "DONE\n\nTask completed successfully." if is_done else (
        f"Working on step {step + 1}...\n\n"
        "Continuing to process the task. More work needed."
    )

    return {
        "type": "result",
        "subtype": "success",
        "cost_usd": round((input_tokens * 0.000003 + output_tokens * 0.000015), 6),
        "is_error": False,
        "duration_ms": 2500 + (step * 500),
        "duration_api_ms": 2000 + (step * 400),
        "num_turns": step + 1,
        "result": result_text,
        "session_id": session_id or "test-session",
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
        "total_cost_usd": round((input_tokens * 0.000003 + output_tokens * 0.000015) * (step + 1), 6),
    }


def load_response(session_id: str | None, step: int, prompt: str) -> dict[str, Any]:
    """Load or generate response for this session/step."""
    response_file = find_response_file(session_id, step)

    if response_file:
        try:
            response = json.loads(response_file.read_text())
            # Always override session_id with actual session
            if session_id:
                response["session_id"] = session_id
            return response
        except (json.JSONDecodeError, OSError) as e:
            print(f"Error loading {response_file}: {e}", file=sys.stderr)

    # Generate default response
    return create_default_response(prompt, session_id, step)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Fake Claude CLI for testing")
    parser.add_argument("--print", action="store_true", dest="print_mode",
                       help="Print mode (required)")
    parser.add_argument("--output-format", choices=["json", "text"], default="text",
                       help="Output format")
    parser.add_argument("-p", "--prompt", type=str,
                       help="The prompt to process")
    parser.add_argument("--session-id", type=str,
                       help="Session ID for new session")
    parser.add_argument("--resume", type=str,
                       help="Resume existing session")

    args = parser.parse_args()

    if not args.print_mode:
        print("Error: --print is required", file=sys.stderr)
        return 1

    if not args.prompt:
        print("Error: -p/--prompt is required", file=sys.stderr)
        return 1

    # Determine session ID
    session_id = args.session_id or args.resume
    is_new_session = args.session_id is not None

    # Get step number
    if is_new_session and session_id:
        reset_session(session_id)

    step = get_next_step(session_id) if session_id else 0

    # Load or generate response
    response = load_response(session_id, step, args.prompt)

    # Output based on format
    if args.output_format == "json":
        print(json.dumps(response, indent=2))
    else:
        # Text mode - just print the result
        print(response.get("result", ""))

    return 0


if __name__ == "__main__":
    sys.exit(main())
