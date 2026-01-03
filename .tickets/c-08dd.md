---
id: c-08dd
status: open
deps: [c-986d]
created: 2026-01-03T18:10:46Z
type: task
priority: 2
assignee: JR Tipton
---
# Create fake_claude.py script for e2e tests

Create tests/fixtures/fake_claude.py - executable script that mimics claude --print --output-format json. Read canned responses from tests/fixtures/responses/*.json. Support multi-step workflows ending in DONE. Include realistic token usage in output.

