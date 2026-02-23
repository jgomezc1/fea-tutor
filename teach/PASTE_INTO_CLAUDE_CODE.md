# Interactive Teaching Session

## File
- `teach.py` → project root `teach.py`

## Setup
```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-...
```

## Integration

### Step 1: Copy `teach.py` to project root.

### Step 2: Verify it works
```bash
python teach.py --list-nodes         # should show all 18 nodes
python teach.py --node assembly      # jump into a live teaching session
```

### Step 3: Update CLAUDE.md
```
## Interactive Teaching
- teach.py: Live teaching sessions with Claude as teacher + evaluator
- Usage: python teach.py --node <node> [--state <state>] [--student <id>]
- Commands during session: /skip /repeat /state /node <n> /quit
- Requires ANTHROPIC_API_KEY
```
