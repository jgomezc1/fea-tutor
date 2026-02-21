# FEA Tutor — Cognitive Apprenticeship Model Agent

## Project Overview

An AI-powered pedagogical agent that teaches Finite Element Analysis (FEA) using the Cognitive Apprenticeship Model (CAM). Unlike a typical chatbot, this system has a **pedagogical state machine** that decides *how* to teach based on the student's evolving understanding.

The key differentiator: the agent evaluates student responses for **physical reasoning** (specifically equilibrium arguments), not just correct answers. A student who gets the right stiffness matrix by memorizing the pattern is NOT considered to have mastered the concept.

## Architecture: Three Components

### 1. Orchestrator (`core/orchestrator.py`) — Pure Python, NO LLM
The deterministic state machine. It:
- Maintains pedagogical state: `ASSESS_PRIOR → MODEL → GUIDED_PRACTICE → DIAGNOSE → [REMEDIATE or FADE] → ASSESS_MASTERY → [ADVANCE or RECYCLE]`
- Manages the student model (mastery probabilities, misconceptions, scaffolding level)
- Decides state transitions based on evaluator output
- Constructs context dicts for teacher and evaluator agents
- Selects problems from the bank based on state and student progress
- **Never calls an LLM** — all logic is deterministic Python

### 2. Teacher Agent (`agents/teacher.py`) — Student-Facing LLM
Receives context from the orchestrator and generates the teaching response. It:
- Has CAM technique-specific prompt templates (modeling, coaching, articulation, remediation, exploration, transition)
- Does NOT make state transition decisions
- Is transparent about assessment ("I want to hear equilibrium reasoning before we move on")
- Adapts tone and depth to scaffolding level
- Uses expert narrations from the problem bank during MODEL state

### 3. Evaluator Agent (`agents/evaluator.py`) — Analytical LLM (Never Student-Facing)
A separate, focused LLM call that classifies student responses. It:
- Outputs structured JSON with: procedural correctness, conceptual depth, misconception detection, physical reasoning detection
- Has no personality — purely analytical
- Is separate from the teacher to prevent pedagogical concerns from biasing evaluation
- Uses the physical reasoning markers from the problem bank for classification

## Key Design Decisions

### Conservative Assessment Policy
- The agent ONLY advances when: P(mastery) >= 0.90 AND level-0 assessment passed AND no unresolved misconceptions
- **Mastery override (cold Level-0 pass):** When a student uses `/skip` to jump directly to ASSESS_MASTERY and passes with all four criteria (procedural=correct, conceptual=deep, physical_reasoning=true, confidence=high), the orchestrator advances regardless of the BKT p_mastery value. Rationale: a perfect score on an unseen problem with zero scaffolding and no prior coaching is the strongest possible direct evidence of mastery — stronger than any statistical accumulation. The BKT estimate is floored to the mastery threshold to keep the student model consistent. Unresolved misconceptions still block advancement even under the override.
- A correct answer with mechanical reasoning (no physical reasoning) is NOT mastery
- The agent always probes reasoning before accepting an answer — articulation is mandatory
- When confidence is low, the agent probes further rather than advancing

### Physical Reasoning as the Key Marker
- The fundamental test of understanding is whether the student uses **equilibrium arguments**
- "The off-diagonal is -k because that's the pattern" → MECHANICAL (not mastery)
- "The off-diagonal is -k because displacing node 1 causes the spring to pull node 2 in the opposite direction" → DEEP (mastery candidate)
- Stumbling while constructing physical reasoning is a POSITIVE signal

### Scaffolding Decay
- Level 3 (heavy): Agent provides most of the solution, student fills a gap
- Level 2 (moderate): Agent provides a strategic hint
- Level 1 (light): Problem statement only, no hints
- Level 0 (none): Assessment mode — no assistance
- Student must succeed at Level 1 before attempting Level 0
- Failure increases scaffolding; success decreases it

### Student Model Persistence
- Student state is saved to JSON after every interaction
- Includes: mastery probabilities (BKT), misconception history, LO status, scaffolding history
- Sessions can be resumed with `--resume`

## Data Files (`data/`)

### `curriculum_graph.json`
DAG of concept nodes with edges representing prerequisites. Each node has:
- Learning objectives (tagged with Bloom's level)
- Known misconceptions with descriptions
- Physical reasoning markers (positive and negative indicators)

All 6 nodes are fully implemented with learning objectives, misconceptions, and physical reasoning markers:
1. `element_stiffness` — Element stiffness matrix derivation
2. `local_global_dofs` — Local vs global DOF mapping
3. `assembly` — Global stiffness matrix assembly
4. `boundary_conditions` — Applying boundary conditions
5. `solution` — Solving and back-substitution
6. `verification` — Verification and interpretation of results

### `problem_bank.json`
Curated problems with full metadata. Each problem has:
- Solution steps with physical reasoning annotations
- Expert narration (for MODEL state)
- Scaffolded versions at levels 1-3
- Evaluation markers (positive/negative physical reasoning indicators)
- Target LOs and misconceptions

18 problems total (3 per node), each with full metadata:
- `ES_A`, `ES_B`, `ES_C`: Element stiffness matrix (derive, interpret physically, singularity/rigid body modes)
- `DOF_A`, `DOF_B`, `DOF_C`: Local vs global DOFs (series spring mapping, relabeling nodes, branching topology)
- `ASM_A`, `ASM_B`, `ASM_C`: Assembly (three-spring series assembly, overlapping contributions interpretation, four-node topology)
- `BC_A`, `BC_B`, `BC_C`: Boundary conditions (fixed DOF elimination, prescribed displacement, mixed BCs)
- `SOL_A`, `SOL_B`, `SOL_C`: Solution (partition and solve, back-substitution for reactions, alternate configuration)
- `VER_A`, `VER_B`, `VER_C`: Verification (equilibrium check, energy-based verification, complete results interpretation)

### `student_model_template.json`
Template for new student state. Includes BKT parameters:
- `p_transit`: 0.2 (probability of learning from one interaction)
- `p_slip`: 0.1 (probability of error despite mastery)
- `p_guess`: 0.05 (probability of correct answer without mastery — low for open-ended FEA)
- `mastery_threshold`: 0.90

## Running the Application

```bash
# New session
python main.py --debug

# Named student
python main.py --student-id alice --debug

# Resume session
python main.py --resume data/student_alice.json --debug
```

### In-Session Commands
- `/status` — Show current state, mastery, misconceptions
- `/skip` — Jump to mastery assessment
- `/repeat` — Return to modeling
- `/debug` — Toggle debug output
- `/quit` — Save and exit

## Development Priorities

### Immediate (Vertical Slice Validation)
1. Test the full CAM cycle for `element_stiffness` (all 3 problems)
2. Validate evaluator accuracy — does it correctly distinguish deep vs mechanical?
3. Test state transitions — does the orchestrator make the right decisions?
4. Validate scaffolding decay — does the fading feel natural?
5. Test the ADVANCE transition to `local_global_dofs`

### Near-Term Improvements
- Add conversation summarization for long sessions (context window management)
- Implement student intent detection (is the student asking a question, submitting work, or requesting help?)
- Add code execution capability so the agent can run and check student computations
- Improve problem selection logic (avoid repeating the same problem)
- Add session analytics / learning trajectory visualization

### Future Expansion
- Integrate PDF class notes as a RAG knowledge source for deeper explanations
- Integrate GitHub notebooks for computational demonstrations
- Add multi-session memory and longitudinal tracking
- Build a Streamlit or web UI

## Code Conventions
- Type hints on all function signatures
- Docstrings on all public functions
- JSON for all data persistence (upgrade to SQLite if needed later)
- The orchestrator is the single source of truth for state — agents never modify state directly
- Evaluator output must always be valid JSON with the required fields
- Teacher never reveals solution steps directly during GUIDED_PRACTICE

## Common Pitfalls to Watch For
- **Evaluator returning non-JSON**: The evaluator sometimes wraps output in markdown code blocks. The `evaluate()` function strips these, but watch for edge cases.
- **Conversation history growing too large**: Currently capped at 40 messages. May need summarization for long sessions.
- **State machine loops**: A student who keeps getting `mechanical` evaluation can loop between GUIDED_PRACTICE and DIAGNOSE. The orchestrator should eventually escalate to re-modeling.
- **Problem repetition**: The current problem selection is simple (modulo rotation). Students may see the same problem twice in practice.
- **Teacher leaking solutions**: The teacher has the full solution in its system prompt for reference. The prompt explicitly says not to reveal it, but this should be monitored.
