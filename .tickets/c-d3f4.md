---
id: c-d3f4
status: closed
deps: []
created: 2026-01-03T19:04:50Z
type: task
priority: 2
assignee: JR Tipton
---
# Add test: verify ticket close is committed with merge

When complete_task() runs, tk close modifies .tickets/<id>.md. Ensure this change is properly committed as part of the merge flow, not left as uncommitted changes in main. Add e2e test that verifies: after agent completes and merges, .tickets/<id>.md shows status=closed AND is committed (not dirty).

