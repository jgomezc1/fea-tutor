# Automated Self-Test

## Overview
Drives the full tutor pipeline with pre-written student responses and REAL Claude API evaluator calls. No human interaction. Verifies: evaluator classifications, state transitions, LO updates, mastery progression, misconception tracking, scaffolding dynamics, persistence, and multi-node advancement.

5 scenarios: quick (2 nodes), misconception handling, scaffolding dynamics, full Module 1 (6 nodes), persistence save/resume.

## File
- `selftest.py` → project root `selftest.py`

## Setup
Requires `teach.py` already integrated (uses `build_evaluator_prompt` from it).

## Integration

### Step 1: Copy `selftest.py` to project root.

### Step 2: Fix the import
`selftest.py` imports `build_evaluator_prompt` from `teach.py`. If teach.py is at project root, add this near the top of selftest.py:
```python
sys.path.insert(0, os.path.dirname(__file__))
```

### Step 3: Test without API calls first
```bash
python selftest.py --dry-run                    # no API calls, just checks the plan
python selftest.py --dry-run --all              # all scenarios dry run
```

### Step 4: Run with real API calls
```bash
python selftest.py --scenario quick             # ~10 API calls, ~30 seconds
python selftest.py --scenario misconception     # ~5 API calls
python selftest.py --scenario scaffolding       # ~5 API calls
python selftest.py --scenario persistence       # ~10 API calls
python selftest.py --scenario full              # ~40 API calls, full Module 1
python selftest.py --all                        # everything
```

### Step 5: Update CLAUDE.md
```
## Self-Test
- selftest.py: Automated end-to-end testing with real Claude API evaluator calls
- 5 scenarios: quick, misconception, scaffolding, full, persistence
- Pre-written student responses calibrated to known quality levels
- Usage: python selftest.py --scenario quick (or --all, or --dry-run)
```
