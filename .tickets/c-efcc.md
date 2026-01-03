---
id: c-efcc
status: closed
deps: []
created: 2026-01-03T18:49:59Z
type: task
priority: 2
assignee: JR Tipton
---
# Add live dashboard mode with -l flag

Add --live / -l flag to dashboard command for auto-refreshing htop-style view. Use Rich Live display, refresh every 2 seconds, press 'q' to exit back to REPL. Extract dashboard rendering to reusable render_dashboard() function. Modify cmd_dashboard in cli.py.

