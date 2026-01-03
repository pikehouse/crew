#!/usr/bin/env python3
"""Generate test tickets programmatically for e2e testing.

This script creates tickets in the .tickets/ directory with configurable
properties for testing various scenarios.

Usage:
    python generate_tickets.py                    # Reset to default tickets
    python generate_tickets.py --clean            # Remove all tickets
    python generate_tickets.py --count 10         # Generate 10 tickets
    python generate_tickets.py --scenario chain   # Generate specific scenario
"""

from __future__ import annotations

import argparse
import random
import shutil
import string
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


TICKETS_DIR = Path(__file__).parent / ".tickets"

# Default ticket templates
DEFAULT_TICKETS = [
    {
        "id": "t-0001",
        "status": "done",
        "deps": [],
        "type": "task",
        "priority": 1,
        "assignee": "test-user",
        "title": "Initialize project structure",
        "body": "Set up the basic project structure with README and sample code.",
    },
    {
        "id": "t-0002",
        "status": "open",
        "deps": ["t-0001"],
        "type": "task",
        "priority": 2,
        "assignee": "test-user",
        "title": "Add unit tests",
        "body": "Create unit tests for the sample module functions.",
    },
    {
        "id": "t-0003",
        "status": "open",
        "deps": ["t-0001"],
        "type": "task",
        "priority": 3,
        "assignee": "",
        "title": "Add documentation",
        "body": "Write documentation for the sample module API.",
    },
    {
        "id": "t-0004",
        "status": "open",
        "deps": ["t-0002", "t-0003"],
        "type": "task",
        "priority": 2,
        "assignee": "",
        "title": "Release v0.1.0",
        "body": "Prepare and tag the first release after tests and docs are complete.",
    },
    {
        "id": "t-0005",
        "status": "blocked",
        "deps": ["t-0004"],
        "type": "feature",
        "priority": 1,
        "assignee": "",
        "title": "Add CLI interface",
        "body": "Add a command-line interface to the sample module.\n\nThis is blocked until v0.1.0 is released.",
    },
]


def generate_ticket_id(index: int) -> str:
    """Generate a ticket ID from an index."""
    return f"t-{index:04d}"


def generate_random_id() -> str:
    """Generate a random 4-character hex ticket ID."""
    return f"t-{''.join(random.choices(string.hexdigits.lower()[:16], k=4))}"


def format_ticket(ticket: dict[str, Any], created: datetime | None = None) -> str:
    """Format a ticket dictionary as markdown with YAML frontmatter."""
    if created is None:
        created = datetime.utcnow()

    deps_str = str(ticket.get("deps", [])).replace("'", '"')
    assignee = ticket.get("assignee", "")

    return f"""---
id: {ticket["id"]}
status: {ticket.get("status", "open")}
deps: {deps_str}
created: {created.strftime("%Y-%m-%dT%H:%M:%SZ")}
type: {ticket.get("type", "task")}
priority: {ticket.get("priority", 2)}
assignee: {assignee}
---
# {ticket["title"]}

{ticket.get("body", "")}
"""


def write_ticket(ticket: dict[str, Any], created: datetime | None = None) -> Path:
    """Write a ticket to the .tickets directory."""
    TICKETS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = TICKETS_DIR / f"{ticket['id']}.md"
    filepath.write_text(format_ticket(ticket, created))
    return filepath


def clean_tickets() -> int:
    """Remove all tickets from the .tickets directory."""
    if not TICKETS_DIR.exists():
        return 0
    count = 0
    for f in TICKETS_DIR.glob("*.md"):
        f.unlink()
        count += 1
    return count


def reset_to_defaults() -> list[Path]:
    """Reset .tickets to the default set of tickets."""
    clean_tickets()
    paths = []
    base_time = datetime(2026, 1, 1, 10, 0, 0)
    for i, ticket in enumerate(DEFAULT_TICKETS):
        created = base_time + timedelta(hours=i)
        paths.append(write_ticket(ticket, created))
    return paths


def generate_random_tickets(count: int, with_deps: bool = True) -> list[Path]:
    """Generate random tickets for testing."""
    clean_tickets()
    paths = []
    base_time = datetime.utcnow() - timedelta(days=count)
    statuses = ["open", "open", "open", "done", "blocked"]
    types = ["task", "task", "feature", "bug"]
    titles = [
        "Implement feature X",
        "Fix bug in module Y",
        "Add tests for Z",
        "Refactor component A",
        "Update documentation",
        "Improve performance",
        "Add error handling",
        "Create API endpoint",
    ]

    for i in range(1, count + 1):
        ticket_id = generate_ticket_id(i)

        # Generate dependencies (only to previous tickets)
        deps = []
        if with_deps and i > 1:
            # 50% chance of having deps
            if random.random() > 0.5:
                num_deps = random.randint(1, min(2, i - 1))
                dep_indices = random.sample(range(1, i), num_deps)
                deps = [generate_ticket_id(j) for j in dep_indices]

        ticket = {
            "id": ticket_id,
            "status": random.choice(statuses),
            "deps": deps,
            "type": random.choice(types),
            "priority": random.randint(1, 3),
            "assignee": random.choice(["", "", "test-user", "agent-1"]),
            "title": random.choice(titles),
            "body": f"Auto-generated ticket {i} for testing.",
        }
        created = base_time + timedelta(hours=i)
        paths.append(write_ticket(ticket, created))

    return paths


def generate_chain_scenario(length: int = 5) -> list[Path]:
    """Generate a linear chain of dependent tickets."""
    clean_tickets()
    paths = []
    base_time = datetime.utcnow()

    for i in range(1, length + 1):
        ticket_id = generate_ticket_id(i)
        deps = [generate_ticket_id(i - 1)] if i > 1 else []
        status = "done" if i == 1 else "open"

        ticket = {
            "id": ticket_id,
            "status": status,
            "deps": deps,
            "type": "task",
            "priority": 2,
            "assignee": "",
            "title": f"Step {i} of {length}",
            "body": f"This is step {i} in a chain of {length} tickets.",
        }
        paths.append(write_ticket(ticket, base_time + timedelta(hours=i)))

    return paths


def generate_parallel_scenario(count: int = 4) -> list[Path]:
    """Generate parallel independent tickets with a common dependency."""
    clean_tickets()
    paths = []
    base_time = datetime.utcnow()

    # Create root ticket
    root = {
        "id": "t-0001",
        "status": "done",
        "deps": [],
        "type": "task",
        "priority": 1,
        "assignee": "test-user",
        "title": "Setup",
        "body": "Initial setup task.",
    }
    paths.append(write_ticket(root, base_time))

    # Create parallel tickets
    for i in range(2, count + 2):
        ticket = {
            "id": generate_ticket_id(i),
            "status": "open",
            "deps": ["t-0001"],
            "type": "task",
            "priority": 2,
            "assignee": "",
            "title": f"Parallel task {i - 1}",
            "body": f"This is parallel task {i - 1} that depends on setup.",
        }
        paths.append(write_ticket(ticket, base_time + timedelta(hours=i)))

    # Create final convergent ticket
    final = {
        "id": generate_ticket_id(count + 2),
        "status": "open",
        "deps": [generate_ticket_id(i) for i in range(2, count + 2)],
        "type": "task",
        "priority": 1,
        "assignee": "",
        "title": "Final convergent task",
        "body": "This task depends on all parallel tasks.",
    }
    paths.append(write_ticket(final, base_time + timedelta(hours=count + 3)))

    return paths


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate test tickets for e2e testing"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove all tickets from .tickets/",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of tickets to generate (default: 5)",
    )
    parser.add_argument(
        "--scenario",
        choices=["default", "random", "chain", "parallel"],
        default="default",
        help="Ticket generation scenario (default: default)",
    )
    parser.add_argument(
        "--no-deps",
        action="store_true",
        help="Generate tickets without dependencies (for random scenario)",
    )

    args = parser.parse_args()

    if args.clean:
        removed = clean_tickets()
        print(f"Removed {removed} ticket(s)")
        return 0

    if args.scenario == "default":
        paths = reset_to_defaults()
        print(f"Reset to {len(paths)} default ticket(s)")
    elif args.scenario == "random":
        paths = generate_random_tickets(args.count, with_deps=not args.no_deps)
        print(f"Generated {len(paths)} random ticket(s)")
    elif args.scenario == "chain":
        paths = generate_chain_scenario(args.count)
        print(f"Generated {len(paths)} chained ticket(s)")
    elif args.scenario == "parallel":
        paths = generate_parallel_scenario(args.count)
        print(f"Generated {len(paths)} ticket(s) in parallel scenario")

    for p in paths:
        print(f"  - {p.name}")

    return 0


if __name__ == "__main__":
    exit(main())
