# crew

Multi-agent orchestrator for Claude Code.

## Overview

`crew` orchestrates multiple Claude Code agents working on the same codebase. Each agent runs in its own git worktree, with its own session, working on assigned tasks.

```
┌─────────────────────────────────────────────────────────────┐
│                        crew                                  │
├─────────────────────────────────────────────────────────────┤
│  REPL UI          State Manager       Agent Runner          │
│  - prompt         - agents.json       - spawn/step/cleanup  │
│  - commands       - ticket ops (tk)   - claude --print -r   │
│  - display        - session IDs       - output capture      │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
         agents/alpha    agents/beta    agents/gamma
         (worktree)      (worktree)     (worktree)
```

## Installation

```bash
cd /path/to/crew
pip install -e .
```

## Prerequisites

- Python 3.11+
- Claude Code CLI (`claude`)
- `ticket` (`tk`) for issue tracking
- Git

## Usage

```bash
cd your-project
crew
```

### Commands

| Command | Description |
|---------|-------------|
| `status`, `s` | Show all agents and queue |
| `spawn <name> [task]` | Create agent on task (or next ready) |
| `step <name>` | Run one step for agent |
| `run` | Continuously step all agents |
| `peek <name>` | Show agent's recent output |
| `logs <name>` | Show log directory |
| `kill <name>` | Stop agent, keep worktree |
| `cleanup <name>` | Remove agent and worktree |
| `merge <name>` | Merge agent's branch to main |
| `ready`, `r` | Show ready tickets |
| `new <title>` | Create ticket |
| `assign <name> <id>` | Assign ticket to agent |
| `help` | Show commands |
| `quit`, `q` | Exit |

## How It Works

1. **Spawn**: Creates a git worktree + writes CLAUDE.md with task instructions
2. **Step**: Runs `claude --print --resume <session>` to continue agent's work
3. **Done**: When agent outputs "DONE", auto-merges branch to main
4. **Logs**: All Claude output logged to `.crew/logs/<agent>/`

## File Structure

```
your-project/
├── .crew/
│   ├── state.json       # Agent registry
│   └── logs/
│       └── alpha/
│           ├── 001-init.log
│           └── 002-step.log
├── agents/              # Git worktrees
│   └── alpha/
│       └── CLAUDE.md
└── .tickets/            # ticket storage
```

## Example Session

```
$ crew

╭─────────────────────────────────────╮
│ crew — multi-agent orchestrator     │
╰─────────────────────────────────────╯

crew> spawn alpha mp-auth
✓ Created agent alpha on mp-auth

[1 active] crew> status
Agents:
  ● alpha: mp-auth (working, 1 steps, 2m)

Ready Work:
  mp-test [P2] blocked on mp-auth

[1 active] crew> run
→ Stepped alpha (step 2)
→ Stepped alpha (step 3)
✓ alpha completed mp-auth
✓ Merged agent/alpha to main

crew> status
No agents.

Ready Work:
  mp-test [P2] - Write auth tests

crew>
```
