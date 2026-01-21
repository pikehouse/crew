"""Fault injection tests for crew state management.

This package provides comprehensive tests for verifying that crew's state
management is resilient to interruptions and failures at any point in the
agent lifecycle.

Key components:
- invariants.py: State invariant checks
- snapshots.py: Snapshot/restore for before/after comparison
- test_*.py: Interrupt tests for each operation
"""
