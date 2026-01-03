---
id: c-1f7f
status: closed
deps: []
created: 2026-01-03T18:10:45Z
type: task
priority: 2
assignee: JR Tipton
---
# Add token tracking fields to Agent dataclass

Add total_input_tokens, total_output_tokens, total_cost_usd fields to Agent in crew/agent.py. These will accumulate usage across steps.

