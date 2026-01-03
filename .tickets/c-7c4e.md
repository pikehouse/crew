---
id: c-7c4e
status: closed
deps: []
created: 2026-01-03T19:00:50Z
type: task
priority: 2
assignee: JR Tipton
---
# Add log rotation - prune logs older than 10 days

Logs accumulate forever in .crew/logs/. On startup, delete log files older than 10 days. Could also add --keep-logs flag to clean command to preserve vs delete.

