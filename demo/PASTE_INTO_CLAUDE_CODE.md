# Feature Demo Runner

## Overview
A single script that showcases every tutor capability independently. No Claude API calls. No student interaction. Runs in seconds.

## File
- `demo.py` → project root `demo.py`

## Integration

### Step 1: Copy `demo.py` to the project root.

### Step 2: Test it
```bash
python demo.py              # list demos
python demo.py curriculum   # just the curriculum graph
python demo.py executor     # just the code executor
python demo.py all          # everything
```

### Step 3: Fix any import paths
The script imports from `core.orchestrator`, `core.notebook_knowledge`, `core.notebook_executor`, `core.spring_agent_bridge`, `core.persistence_manager`. Adjust if your package structure differs.

### Step 4: Update CLAUDE.md
```
## Demo Runner
- demo.py: Feature showcase for presentations. No API calls needed.
- Usage: python demo.py [curriculum|state|notebook|executor|solver|persistence|context|all]
```
