---
id: c-75bb
status: closed
deps: []
created: 2026-01-03T18:33:28Z
type: task
priority: 2
assignee: JR Tipton
---
# Remove step command from REPL

Remove cmd_step() and its registration from cli.py. Update display.py help text. The step command is not useful in practice - users should just use run instead.

