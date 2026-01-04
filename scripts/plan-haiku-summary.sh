#!/bin/bash
# Plan: Haiku-powered Agent Work Summary
# Creates tickets for implementing diff summarization with Claude Haiku

set -e

echo "Creating tickets for Haiku summary feature..."

# Core helper
T1=$(tk create "Add summarize_diff() helper to runner.py" -p 2)
echo "Created $T1: summarize_diff helper"

# CLI command
T2=$(tk create "Add cmd_summarize command to CLI" -p 2 -d "$T1")
echo "Created $T2: summarize command (depends on $T1)"

# Command routing
T3=$(tk create "Add summarize/sum command routing in handle_command" -p 2 -d "$T2")
echo "Created $T3: command routing (depends on $T2)"

# Dashboard integration
T4=$(tk create "Integrate Haiku summary into dashboard agent panels" -p 3 -d "$T1")
echo "Created $T4: dashboard integration (depends on $T1)"

# Optional: make dashboard summary opt-in
T5=$(tk create "Add -s flag to dashboard for optional summaries" -p 3 -d "$T4")
echo "Created $T5: optional summary flag (depends on $T4)"

echo ""
echo "Created 5 tickets:"
echo "  $T1 - summarize_diff helper"
echo "  $T2 - cmd_summarize command"
echo "  $T3 - command routing"
echo "  $T4 - dashboard integration"
echo "  $T5 - optional -s flag"
echo ""
echo "Run 'tk queue' to see the ticket queue"
