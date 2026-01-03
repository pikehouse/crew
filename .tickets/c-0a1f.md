---
id: c-0a1f
status: closed
deps: [c-f038]
created: 2026-01-03T18:33:27Z
type: task
priority: 2
assignee: JR Tipton
---
# Add PID file locking for single-instance enforcement

On startup, create .crew/crew.pid with current PID. Check if existing PID file exists and process is running. If another crew is running, enter read-only mode: can view dashboard/peek but cannot spawn/run/kill/modify. Show [read-only] in prompt. Remove PID file on clean exit. Detect stale PID on startup.

