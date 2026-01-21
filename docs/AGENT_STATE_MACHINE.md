# Crew Agent State Machine

This document describes how Crew orchestrates and manages autonomous coding agents.

## Overview

Crew manages multiple AI coding agents that work in parallel on tasks from a ticket queue. Each agent operates in an isolated git worktree and progresses through a well-defined state machine until its task is complete.

---

## Agent States

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│  IDLE   │────▶│  READY  │────▶│ WORKING │────▶│  DONE   │
│    ○    │     │    ◐    │     │    ●    │     │    ✓    │
└─────────┘     └─────────┘     └─────────┘     └─────────┘
                                     │
                                     ▼
                               ┌─────────┐
                               │  STUCK  │
                               │    !    │
                               └─────────┘
```

| State | Symbol | Description | Active? |
|-------|--------|-------------|---------|
| `idle` | ○ | Agent exists but has no assigned task | Yes |
| `ready` | ◐ | Task assigned, worktree created, waiting to step | Yes |
| `working` | ● | Agent is actively executing steps via Claude | Yes |
| `done` | ✓ | Agent output "DONE", awaiting test/merge | No |
| `stuck` | ! | 5+ consecutive errors, requires intervention | No |

**Active** means the agent can be stepped by the background runner.

---

## Complete State Machine Diagram

```
                              ┌──────────────────┐
                              │   spawn_worker   │
                              │                  │
                              │  Creates agent   │
                              │  with no task    │
                              └────────┬─────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────┐
                    │              IDLE                    │
                    │                                      │
                    │  • session = ""                      │
                    │  • worktree = None                   │
                    │  • branch = ""                       │
                    │  • task = None                       │
                    │  • step_count = 0                    │
                    └─────────────────┬───────────────────┘
                                      │
                          assign_task(agent, task_id)
                                      │
                    ┌─────────────────▼───────────────────┐
                    │              READY                   │
                    │                                      │
                    │  • Creates worktree: agents/<name>   │
                    │  • Creates branch: agent/<name>      │
                    │  • Generates session_id (UUID)       │
                    │  • Writes CLAUDE.md with task        │
                    │  • status = "ready"                  │
                    └─────────────────┬───────────────────┘
                                      │
                              step_agent() [first]
                              INIT_PROMPT + new session
                                      │
                    ┌─────────────────▼───────────────────┐
                    │             WORKING                  │
         ┌─────────▶│                                      │
         │          │  • status = "working"                │
         │          │  • step_count++                      │
         │          │  • Runs Claude in worktree           │
         │          │  • Accumulates tokens/cost           │
         │          └──────┬──────────────┬───────────────┘
         │                 │              │
         │     No "DONE"   │              │  Output contains "DONE"
         │     in output   │              │  OR step_count >= 20
         │                 │              │
         │    step_agent() │              │
         │    STEP_PROMPT  │              ▼
         │    resume       │  ┌───────────────────────────┐
         └─────────────────┘  │           DONE            │
                              │                           │
                              │  • Tests run automatically │
                              │  • If tests fail → WORKING │
                              │  • If tests pass → merge   │
                              └─────────────┬─────────────┘
                                            │
                              ┌─────────────▼─────────────┐
                              │      complete_task()      │
                              │                           │
                              │  1. Remove worktree       │
                              │  2. Checkout main         │
                              │  3. Merge --no-ff         │
                              │  4. Resolve conflicts     │
                              │  5. Delete branch         │
                              │  6. Close ticket (tk)     │
                              │  7. Reset agent → IDLE    │
                              └───────────────────────────┘

                    ┌─────────────────────────────────────┐
                    │              STUCK                   │
                    │                                      │
                    │  Entered when:                       │
                    │  • 5+ consecutive step errors        │
                    │  • Exponential backoff exhausted     │
                    │                                      │
                    │  Auto-recovery (on next run loop):   │
                    │  • Worktree dirty → WORKING          │
                    │  • Worktree clean → READY            │
                    │  • No worktree → IDLE                │
                    │                                      │
                    │  Manual recovery:                    │
                    │  • `kill <name>` - reset to idle     │
                    │  • `cleanup <name>` - remove state   │
                    └─────────────────────────────────────┘
```

---

## State Transitions

### IDLE → READY (`assign_task`)

```python
assign_task(agent, task_id, state)
```

1. Fetches task details via `tk show <task_id>`
2. Creates git worktree at `agents/<agent_name>-<task_id>/`
3. Creates branch `agent/<agent_name>-<task_id>`
4. Generates new `session_id` (UUID)
5. Writes `CLAUDE.md` with task description and rules
6. Sets `status = "ready"`

**Invariants after transition:**
- Worktree directory exists on disk
- Branch is checked out in worktree
- `CLAUDE.md` contains task instructions
- Session ID is set

### READY → WORKING (`step_agent`, first step)

```python
step_agent(agent, state)  # step_count == 0
```

1. Sets `status = "working"` immediately
2. Runs Claude with `INIT_PROMPT` and `--session-id <id>`
3. Increments `step_count`
4. Updates `last_step_at` timestamp
5. Accumulates token usage and cost

### WORKING → WORKING (`step_agent`, subsequent steps)

```python
step_agent(agent, state)  # step_count > 0
```

1. Runs Claude with `STEP_PROMPT` and `--resume <session_id>`
2. Increments `step_count`
3. Updates `last_step_at` timestamp
4. Checks output for "DONE" marker

### WORKING → DONE (completion detected)

Transition occurs when:
- Agent output contains the word "DONE" on its own line
- OR `step_count >= 20` (max steps reached)

### DONE → Test Validation

```python
complete_task(agent, state)
```

1. Runs tests in worktree (auto-detects test runner)
2. **If tests fail:** Returns to WORKING with `TEST_FAILURE_PROMPT`
3. **If tests pass:** Proceeds to merge

### DONE → Merge → IDLE

1. Removes worktree (so branch can be merged)
2. Checks out main branch
3. Merges agent branch with `--no-ff`
4. If conflicts: Uses Claude with `MERGE_CONFLICT_PROMPT` to resolve
5. Deletes the agent branch
6. Closes ticket via `tk close <task_id>`
7. Resets all agent fields to idle state

### WORKING → STUCK (error threshold)

Transition occurs when:
- 5 consecutive step errors occur
- Each error triggers exponential backoff: 2, 4, 8, 16, 32 seconds
- After 5th error, agent is marked stuck

---

## Stall Detection

The runner monitors agent output to detect when an agent is stuck without producing new output.

**How it works:**
1. Every 30 seconds, the runner checks working agents
2. For each agent, it hashes the last 20 lines of the session file (`~/.claude/projects/<path>/<session>.jsonl`)
3. If the hash hasn't changed in 5 minutes, the agent is considered stalled

**On stall detection:**
1. Kill the Claude process (SIGTERM)
2. Generate a new session ID
3. Clear hash tracking for the agent
4. Agent will restart on next runner loop iteration

```
┌─────────────────────────────────────────────────────────────┐
│                    STALL DETECTION                           │
└─────────────────────────────────────────────────────────────┘

Every 30 seconds:
     │
     ▼
┌────────────────────────────┐
│ For each working agent:    │
│ Hash last 20 lines of      │
│ session JSONL file         │
└─────────────┬──────────────┘
              │
     ┌────────┴────────┐
     │                 │
Hash changed?     Hash unchanged?
     │                 │
     ▼                 ▼
┌──────────┐    ┌──────────────────┐
│ Update   │    │ Check duration   │
│ tracking │    │ since last change│
└──────────┘    └────────┬─────────┘
                         │
                ┌────────┴────────┐
                │                 │
            < 5 min           >= 5 min
                │                 │
                ▼                 ▼
           ┌────────┐    ┌───────────────┐
           │ Wait   │    │ Kill process  │
           │        │    │ New session   │
           └────────┘    │ Agent restarts│
                         └───────────────┘
```

---

## Stuck Agent Recovery

When an agent enters the "stuck" state (5+ consecutive errors), it is automatically recovered on the next runner loop based on worktree state:

```
┌─────────────────────────────────────────────────────────────┐
│                  STUCK AGENT RECOVERY                        │
└─────────────────────────────────────────────────────────────┘

For each stuck agent:
     │
     ▼
┌────────────────────────────┐
│ Check worktree exists?     │
└─────────────┬──────────────┘
              │
     ┌────────┴────────┐
     │                 │
  No worktree      Worktree exists
     │                 │
     ▼                 ▼
┌──────────┐    ┌──────────────────┐
│ Reset to │    │ Check git status │
│   IDLE   │    │ (uncommitted?)   │
└──────────┘    └────────┬─────────┘
                         │
                ┌────────┴────────┐
                │                 │
            Clean            Dirty
                │                 │
                ▼                 ▼
           ┌────────┐    ┌───────────────┐
           │ Reset  │    │ Reset to      │
           │ to     │    │ WORKING       │
           │ READY  │    │ (continue)    │
           └────────┘    └───────────────┘
```

**Recovery actions:**
- **No worktree**: Reset to IDLE, task returns to queue
- **Clean worktree**: Reset to READY with new session, restart from beginning
- **Dirty worktree**: Reset to WORKING with new session, continue from where it left off

This ensures stuck agents automatically retry without losing their work.

---

## Error Handling & Backoff

```
Error Count │ Backoff Duration │ Action
────────────┼──────────────────┼─────────────────────────
     1      │     2 seconds    │ Log error, queue event
     2      │     4 seconds    │ Log error, queue event
     3      │     8 seconds    │ Log error, queue event
     4      │    16 seconds    │ Log error, queue event
     5      │    32 seconds    │ Set status = "stuck"
    6+      │    60 seconds    │ Max backoff (if manually retried)
```

**Error Recovery:**
- Successful step resets error count to 0
- Session errors trigger new session_id generation
- Timeouts retry once with fresh session
- Stalled agents are killed and restarted with new session

---

## Session Management

```
┌─────────────────────────────────────────────────────────────┐
│                    SESSION LIFECYCLE                         │
└─────────────────────────────────────────────────────────────┘

assign_task()
     │
     ▼
┌────────────────────┐
│ generate_session_id │ ─────▶ UUID format: xxxxxxxx-xxxx-...
└────────────────────┘
     │
     ▼
step_agent() [first]
     │
     ▼
┌────────────────────┐
│ claude --session-id │ ─────▶ Creates new Claude session
└────────────────────┘
     │
     ▼
step_agent() [subsequent]
     │
     ▼
┌────────────────────┐
│ claude --resume     │ ─────▶ Resumes existing session
└────────────────────┘
     │
     ▼
complete_task()
     │
     ▼
┌────────────────────┐
│ session = ""        │ ─────▶ Session cleared on completion
└────────────────────┘
```

**Session Error Recovery:**
- "Session in use" error → Generate new session_id, retry
- Process killed → Session persists, can be resumed on restart

---

## Parallel Execution Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    BackgroundRunner                          │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │               ThreadPoolExecutor                      │   │
│  │                  max_workers=10                       │   │
│  │                                                       │   │
│  │   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │   │
│  │   │ Agent A │ │ Agent B │ │ Agent C │ │   ...   │   │   │
│  │   │ step()  │ │ step()  │ │ step()  │ │         │   │   │
│  │   └─────────┘ └─────────┘ └─────────┘ └─────────┘   │   │
│  │                                                       │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  Loop every ~1 second:                                       │
│  1. _recover_done_agents() - Handle stuck "done" agents      │
│  2. _assign_work() - Poll tk ready, assign to idle agents    │
│  3. _get_working_agents() - Get ready/working agents         │
│  4. _step_one_agent() - Step each in parallel                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Thread Safety:**
- `_stepping` set prevents double-stepping same agent
- `_merge_lock` serializes merge operations
- Events queue for async communication with dashboard

---

## Git Worktree Isolation

```
project/
├── .git/                          # Main repository
├── .crew/
│   └── state.json                 # Agent state persistence
├── .tickets/                      # Task definitions
│   ├── c-abc1.md
│   └── c-def2.md
├── agents/                        # Worktrees (gitignored)
│   ├── A-c-abc1/                  # Agent A's worktree
│   │   ├── .git                   # Worktree link
│   │   ├── CLAUDE.md              # Task instructions
│   │   └── ...                    # Full repo copy
│   └── B-c-def2/                  # Agent B's worktree
│       └── ...
└── src/                           # Main branch code
```

**Benefits:**
- Agents work in complete isolation
- No branch checkout conflicts
- Parallel file modifications
- Easy cleanup (just remove worktree dir)

---

## Recovery & Persistence

### State Persistence

State is saved to `.crew/state.json` after every operation:

```json
{
  "agents": [
    {
      "name": "A",
      "session": "abc-123-...",
      "worktree": "/path/to/agents/A-c-abc1",
      "branch": "agent/A-c-abc1",
      "task": "c-abc1",
      "task_assigned_at": "2024-01-15T10:30:00Z",
      "status": "working",
      "started_at": "2024-01-15T10:00:00Z",
      "step_count": 5,
      "last_step_at": "2024-01-15T10:35:00Z",
      "total_input_tokens": 50000,
      "total_output_tokens": 10000,
      "total_cost_usd": 0.25
    }
  ]
}
```

### Session Recovery on Startup

```
recover_session() runs at CLI startup:

┌────────────────────────────────────────────────────────────┐
│ For each agent in state.json:                               │
├────────────────────────────────────────────────────────────┤
│                                                             │
│ Worktree missing + ready/working?                          │
│   └─▶ Reset to IDLE, keep ticket open                      │
│                                                             │
│ Dirty worktree?                                             │
│   └─▶ Remove worktree, reset to IDLE                       │
│                                                             │
│ Clean worktree + ready/working?                            │
│   └─▶ Check logs for DONE marker                           │
│       ├─ DONE found → run complete_task()                  │
│       └─ No DONE → reset to IDLE                           │
│                                                             │
│ Status = "done" + worktree exists?                         │
│   └─▶ Run complete_task() to finish merge                  │
│                                                             │
│ Status = "done" + worktree missing?                        │
│   └─▶ Offer `merge <name>` command                         │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

---

## CLI Commands by State

| Command | Effect | Valid From States |
|---------|--------|-------------------|
| `spawn [name]` | Create idle agent | N/A (creates new) |
| `spawn <count>` | Create multiple idle agents | N/A |
| `assign <name> <task>` | Assign task to agent | idle |
| `run` | Start background runner | Any |
| `stop` | Stop runner, preserve state | Any |
| `ps` | Show agent status table | Any |
| `peek <name>` | Show recent log output | Any |
| `logs <name>` | Show full logs | Any |
| `kill <name>` | Kill Claude process, reset | working, stuck |
| `cleanup <name>` | Remove worktree, keep state | Any |
| `merge <name>` | Manual merge of branch | done |
| `dashboard` / `d` | Live status view | Any |
| `clean` | Full wipe - remove all | Any |

---

## Orchestration Flow Summary

```
     ┌─────────────────────────────────────────────────────────┐
     │                      tk ready                            │
     │              (External task queue)                       │
     └─────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                    BackgroundRunner                           │
│                                                               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │   RECOVER   │    │   ASSIGN    │    │    STEP     │      │
│  │             │    │             │    │             │      │
│  │ Fix stuck   │    │ idle→ready  │    │ working→    │      │
│  │ "done"      │    │ from tk     │    │ done        │      │
│  │ agents      │    │ queue       │    │             │      │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘      │
│         │                  │                  │              │
│         └──────────────────┼──────────────────┘              │
│                            │                                  │
│                            ▼                                  │
│                   ┌─────────────────┐                        │
│                   │     Events      │                        │
│                   │     Queue       │                        │
│                   └────────┬────────┘                        │
│                            │                                  │
└────────────────────────────┼──────────────────────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    Dashboard    │
                    │                 │
                    │  Live status    │
                    │  Token usage    │
                    │  Cost tracking  │
                    │  Agent summaries│
                    └─────────────────┘
```

---

## Consolidated Stuck Detection

All stuck detection logic is consolidated in `crew/stuck_detection.py`:

```python
@dataclass
class StuckTracker:
    """Tracks stuck detection state for a single agent."""
    error_count: int = 0           # Consecutive errors
    backoff_until: float = 0.0     # Time to wait before retry
    last_output_hash: str | None   # For stall detection
    last_output_change: float | None  # When output last changed

def check_if_stuck(agent, tracker, step_limit=20, error_limit=5, stall_minutes=5):
    """Single function to check all stuck conditions."""
    # Returns StuckReason or None
```

**Three detection mechanisms unified:**

| Mechanism | Trigger | Result |
|-----------|---------|--------|
| Step limit | `step_count >= 20` | Mark as DONE (or STUCK) |
| Error limit | 5 consecutive errors | Mark as STUCK |
| Output stall | No output change for 5 min | Kill process, restart |

**State reconciliation** via `reconcile_agent_state()` in `runner.py`:

```python
def reconcile_agent_state(agent, project_root) -> Literal["idle", "ready", "working", "done"]:
    """Single source of truth for determining agent status."""
    # No worktree → idle
    # Worktree + DONE marker → done
    # Worktree + dirty → working
    # Worktree + clean → ready
```

---

## Testing the State Machine

The state machine is validated by comprehensive **fault injection tests** that verify invariants hold under all conditions.

### Test Categories

```
tests/fault_injection/
├── test_assign.py      # Task assignment interrupts (7 tests)
├── test_complete.py    # Task completion interrupts (10 tests)
├── test_step.py        # Agent stepping interrupts (12 tests)
├── test_recovery.py    # Recovery scenarios (15 tests)
├── test_properties.py  # Property-based random sequences (5 tests)
├── invariants.py       # Invariant checking utilities
└── snapshots.py        # State snapshot utilities
```

### State Invariants

These invariants are checked after every operation:

```python
INVARIANT_CHECKERS = {
    "idle_agent_has_no_worktree",      # idle → worktree = None
    "working_agent_has_worktree",       # working/ready/done → worktree exists
    "no_duplicate_task_assignments",    # Each task → at most one agent
    "session_id_valid",                 # Non-idle agents have session ID
    "worktree_exists_if_assigned",      # Worktree path exists on disk
    "branch_exists_if_working",         # Agent's branch exists in git
    "ticket_closed_only_if_work_in_main", # Ticket closed → work merged
    "state_file_valid",                 # state.json is valid JSON
}
```

### Fault Injection Scenarios

**Assignment interrupts:**
- Interrupt after worktree creation, before state save
- Orphaned worktree cleanup on retry
- State file never corrupted mid-write

**Completion interrupts:**
- Interrupt after worktree removal, before merge
- Interrupt after merge, before ticket close
- Ticket never closed before merge succeeds

**Step interrupts:**
- Timeout preserves agent state
- Session conflict triggers auto-recovery
- Resume after interrupt continues progress

**Property-based testing:**
```python
@given(st.lists(st.sampled_from(["spawn", "assign", "step", "complete"])))
def test_random_operations_maintain_invariants(operations):
    """Any sequence of valid operations maintains all invariants."""
```

### Running the Tests

```bash
# All tests (452 total)
pytest tests/ -v

# Just fault injection tests (53 tests)
pytest tests/fault_injection/ -v

# With coverage
pytest tests/ --cov=crew --cov-report=html
```

---

## Key Design Principles

1. **State-Driven**: Everything is persisted to `state.json`, enabling recovery from any interruption

2. **Isolation via Worktrees**: Git worktrees provide complete isolation without branch conflicts

3. **Idempotent Operations**: Recovery can run multiple times safely

4. **Exponential Backoff**: Prevents error spam while allowing retry

5. **External Task Queue**: Decoupled from task management (uses `tk` CLI)

6. **Parallel Execution**: Thread pool enables multiple agents working simultaneously

7. **Graceful Degradation**: Stuck agents don't block others; partial work is preserved

8. **Observable**: Dashboard, events, and summaries provide visibility into agent progress

9. **Invariant-Tested**: Fault injection tests verify state machine correctness under all conditions
