# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is crew?

crew is a multi-agent orchestrator for Claude Code. It manages multiple Claude Code agents working in parallel on the same codebase, each in their own git worktree with its own session and assigned task.

## Commands

```bash
# Install in dev mode
pip install -e .

# Run crew
crew

# Run tests
pytest                    # all tests
pytest tests/test_cli.py  # single file
pytest -k test_spawn      # by name pattern

# Run e2e tests (slower, uses real git)
pytest tests/e2e/
```

## Architecture

```
crew/
├── cli.py       # REPL interface, command handlers, BackgroundRunner
├── state.py     # State dataclass + JSON persistence (.crew/state.json)
├── agent.py     # Agent dataclass (name, session, worktree, status, etc.)
├── runner.py    # spawn_worker, assign_task, step_agent, complete_task
├── git.py       # Worktree creation/removal, merge, branch ops
├── display.py   # Rich console output (tables, panels, status icons)
├── crew_logging.py  # Log file management for agent output
└── pid.py       # PID file locking for single-instance
```

### Core Flow

1. **spawn_worker(name)** - Creates idle Agent (no worktree yet)
2. **assign_task(agent, task_id)** - Creates worktree + branch, writes CLAUDE.md with task instructions, sets status="ready"
3. **step_agent(agent)** - Runs `claude --print --resume <session>`, parses JSON output, logs to .crew/logs/
4. **complete_task(agent)** - Runs tests, merges branch to main, closes ticket via `tk close`, resets agent to idle

### Agent States

- `idle` - Waiting for work (no worktree)
- `ready` - Has assigned task, ready to step
- `working` - Currently being stepped
- `done` - Said "DONE", ready for merge
- `stuck` - Hit step limit (20)

### Key Patterns

- **BackgroundRunner** (cli.py) - Polls agents in thread pool, auto-assigns ready tickets to idle workers
- **Session management** - Each task gets a fresh UUID session; first step uses `--session-id`, subsequent use `--resume`
- **Completion detection** - Looks for "DONE" on its own line in agent output
- **Test gating** - Tests run before merge; failure sends agent back to working with test output as prompt

### Integration with ticket (tk)

crew uses the external `tk` CLI for ticket management:
- `tk ready` - List unblocked tickets
- `tk show <id>` - Get ticket content for CLAUDE.md
- `tk close <id>` - Mark complete after merge
- `tk create`, `tk dep` - Wrapped by crew's `new` and `dep` commands

### File Structure at Runtime

```
project/
├── .crew/
│   ├── state.json    # Agent registry
│   ├── pid           # Lock file
│   └── logs/<agent>/ # Step logs (001-step.log, etc.)
├── agents/           # Git worktrees
│   └── a-c-xxxx/     # Named: {agent}-{ticket}
│       └── CLAUDE.md # Task instructions
└── .tickets/         # ticket storage (external)
```
