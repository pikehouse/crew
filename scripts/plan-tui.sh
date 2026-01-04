#!/bin/bash
# Plan: Interactive TUI mode for crew
# Run from project root: ./scripts/plan-tui.sh

set -e

echo "Creating TUI feature tickets..."

# Foundation
t1=$(tk create "Add textual dependency to pyproject.toml" | awk '{print $1}')
echo "Created $t1 - Add textual dependency"

t2=$(tk create "Create basic TUI skeleton (crew/tui.py) - Textual app that launches and exits with q" | awk '{print $1}')
tk dep "$t2" "$t1"
echo "Created $t2 - Basic TUI skeleton (depends on $t1)"

t3=$(tk create "Wire 'tui' command in cli.py handle_command to launch TUI" | awk '{print $1}')
tk dep "$t3" "$t2"
echo "Created $t3 - Wire tui command (depends on $t2)"

# Main view components
t4=$(tk create "TUI: Add agents DataTable with columns (name, task, status, steps, cost)" | awk '{print $1}')
tk dep "$t4" "$t2"
echo "Created $t4 - Agents DataTable (depends on $t2)"

t5=$(tk create "TUI: Add ticket tree widget showing deps structure (reuse cmd_queue wave logic)" | awk '{print $1}')
tk dep "$t5" "$t2"
echo "Created $t5 - Ticket tree widget (depends on $t2)"

t6=$(tk create "TUI: Create split layout - tickets tree left, agents table right" | awk '{print $1}')
tk dep "$t6" "$t4"
tk dep "$t6" "$t5"
echo "Created $t6 - Split layout (depends on $t4, $t5)"

# Detail panel
t7=$(tk create "TUI: Add agent detail panel (git status, diff stat, log tail) - updates on agent selection" | awk '{print $1}')
tk dep "$t7" "$t6"
echo "Created $t7 - Agent detail panel (depends on $t6)"

# Zoom mode
t8=$(tk create "TUI: Implement zoom-to-ticket-tree mode (Enter to zoom, q to unzoom)" | awk '{print $1}')
tk dep "$t8" "$t6"
echo "Created $t8 - Zoom ticket tree (depends on $t6)"

# Actions
t9=$(tk create "TUI: Add kill agent action (k hotkey) with confirmation" | awk '{print $1}')
tk dep "$t9" "$t7"
echo "Created $t9 - Kill action (depends on $t7)"

t10=$(tk create "TUI: Add merge agent action (m hotkey) with confirmation" | awk '{print $1}')
tk dep "$t10" "$t7"
echo "Created $t10 - Merge action (depends on $t7)"

# Runner integration
t11=$(tk create "TUI: Integrate BackgroundRunner events - live updates when agents step/complete" | awk '{print $1}')
tk dep "$t11" "$t7"
echo "Created $t11 - Runner integration (depends on $t7)"

# Polish
t12=$(tk create "TUI: Add run/stop hotkeys (r/s) to control BackgroundRunner from TUI" | awk '{print $1}')
tk dep "$t12" "$t11"
echo "Created $t12 - Run/stop hotkeys (depends on $t11)"

t13=$(tk create "TUI: Add help overlay (? hotkey) showing all keybindings" | awk '{print $1}')
tk dep "$t13" "$t12"
echo "Created $t13 - Help overlay (depends on $t12)"

echo ""
echo "Done! Created 13 tickets."
echo ""
echo "Dependency graph:"
echo "  $t1 (textual dep)"
echo "    └─ $t2 (skeleton)"
echo "        ├─ $t3 (wire command)"
echo "        ├─ $t4 (agents table)"
echo "        └─ $t5 (ticket tree)"
echo "            └─ $t6 (split layout)"
echo "                ├─ $t7 (detail panel)"
echo "                │   ├─ $t9 (kill)"
echo "                │   ├─ $t10 (merge)"
echo "                │   └─ $t11 (runner events)"
echo "                │       └─ $t12 (run/stop)"
echo "                │           └─ $t13 (help)"
echo "                └─ $t8 (zoom)"
echo ""
echo "Run 'tk queue' or 'crew' then 'q' to see the full pipeline."
