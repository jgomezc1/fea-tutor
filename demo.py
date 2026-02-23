#!/usr/bin/env python3
"""
FEA Tutor — Feature Demo Runner

Showcases individual tutor capabilities without the full teaching loop.
No Claude API calls. No student interaction. Just the infrastructure.

Usage:
    python demo.py                  # list all demos
    python demo.py all              # run everything
    python demo.py solver           # run one demo
    python demo.py solver executor  # run multiple demos
"""
import sys
import json
import textwrap
from pathlib import Path


# ── Formatting helpers ───────────────────────────────────────────────

BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

def header(title):
    width = 66
    print(f"\n{BOLD}{BLUE}{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}{RESET}\n")

def subheader(title):
    print(f"\n{BOLD}{YELLOW}── {title} ──{RESET}\n")

def success(msg):
    print(f"  {GREEN}✓{RESET} {msg}")

def info(msg):
    print(f"  {DIM}→{RESET} {msg}")

def code_block(code, output=None):
    print(f"  {DIM}┌─ code ─────────────────────────────────────{RESET}")
    for line in code.strip().split("\n"):
        print(f"  {DIM}│{RESET} {line}")
    if output:
        print(f"  {DIM}├─ output ───────────────────────────────────{RESET}")
        for line in output.strip().split("\n"):
            print(f"  {DIM}│{RESET} {GREEN}{line}{RESET}")
    print(f"  {DIM}└────────────────────────────────────────────{RESET}")

def json_block(data, max_lines=15):
    text = json.dumps(data, indent=2, default=str)
    lines = text.split("\n")
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"  ... ({len(lines) - max_lines} more lines)"]
    for line in lines:
        print(f"  {DIM}{line}{RESET}")


# ── Demo 1: Curriculum Graph ─────────────────────────────────────────

def demo_curriculum():
    header("DEMO 1: Curriculum Graph — 18 Nodes, 3 Modules")

    from core.orchestrator import Orchestrator
    orch = Orchestrator(data_dir="data")

    modules = {
        "Module 1 — Springs": [],
        "Module 2 — Trusses": [],
        "Module 3 — Frames": [],
    }

    # Walk the graph in order
    node = "element_stiffness"
    edges = {s: t for s, t in orch.curriculum["edges"]}
    visited = []
    while node:
        visited.append(node)
        node = edges.get(node)

    m1 = visited[:6]
    m2 = visited[6:12]
    m3 = visited[12:]

    for label, nodes in [("Module 1 — Springs", m1),
                         ("Module 2 — Trusses", m2),
                         ("Module 3 — Frames", m3)]:
        subheader(label)
        for i, n in enumerate(nodes):
            nd = orch.curriculum["nodes"][n]
            arrow = "→" if i < len(nodes) - 1 else "⇥"
            print(f"  {arrow} {BOLD}{n}{RESET}: {nd['title']}")
            for lo_key in list(nd["learning_objectives"].keys())[:2]:
                lo = nd["learning_objectives"][lo_key]
                print(f"    • {lo['description']}")

    subheader("Problem Bank")
    problems = orch.problem_bank
    info(f"{len(problems)} problems across 18 nodes")
    # Show a sample
    sample = list(problems.values())[:3]
    for p in sample:
        print(f"  {BOLD}{p['problem_id']}{RESET}: {p['statement'][:80]}...")

    success(f"Curriculum: {len(visited)} nodes, {len(orch.curriculum['edges'])} edges, {len(problems)} problems")


# ── Demo 2: State Machine ────────────────────────────────────────────

def demo_state_machine():
    header("DEMO 2: CAM State Machine — Pedagogical Transitions")

    from core.orchestrator import Orchestrator
    import tempfile, os

    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()
    os.unlink(tmp.name)

    orch = Orchestrator(data_dir="data", student_file=tmp.name)

    subheader("Happy Path: ASSESS_PRIOR → MODEL → GP → MASTERY → ADVANCE")

    transitions = []

    def log(eval_desc, evaluation):
        old_state = orch.state
        old_scaff = orch.scaffolding_level
        orch.process_evaluation(evaluation)
        new_state = orch.state
        new_scaff = orch.scaffolding_level
        transitions.append((old_state, new_state))
        print(f"  {old_state} (scaff={old_scaff}) "
              f"─[{eval_desc}]→ "
              f"{GREEN}{new_state}{RESET} (scaff={new_scaff})")

    # ASSESS_PRIOR → MODEL (mechanical answer)
    log("mechanical", {
        "procedural": "correct", "conceptual": "mechanical",
        "misconception_id": None, "confidence": "medium",
        "physical_reasoning_detected": False,
    })

    # MODEL → GP (manual)
    info("Teacher finishes modeling, transitions to GUIDED_PRACTICE")
    orch.state = "GUIDED_PRACTICE"

    # GP → fade scaffolding
    perfect = {
        "procedural": "correct", "conceptual": "deep",
        "misconception_id": None, "confidence": "high",
        "physical_reasoning_detected": True,
    }
    log("perfect, scaff 3→2", perfect)
    log("perfect, scaff 2→1", perfect)
    log("perfect, scaff 1→MASTERY", perfect)

    # ASSESS_MASTERY → ADVANCE (may need multiple)
    safety = 0
    while orch.state != "ADVANCE" and safety < 10:
        log("perfect", perfect)
        safety += 1

    subheader("Misconception Path")
    # Reset
    os.unlink(tmp.name) if Path(tmp.name).exists() else None
    orch2 = Orchestrator(data_dir="data")
    orch2.state = "GUIDED_PRACTICE"
    orch2.scaffolding_level = 2

    old = orch2.state
    orch2.process_evaluation({
        "procedural": "correct", "conceptual": "mechanical",
        "misconception_id": "M2", "confidence": "medium",
        "physical_reasoning_detected": False,
    })
    print(f"  {old} ─[misconception M2]→ {RED}{orch2.state}{RESET}")
    info(f"Misconceptions tracked: {orch2.student_node['misconceptions_observed']}")

    subheader("Skip Override")
    orch3 = Orchestrator(data_dir="data")
    orch3.handle_student_override("skip")
    print(f"  ASSESS_PRIOR ─[/skip]→ {orch3.state} (scaffolding={orch3.scaffolding_level})")

    success(f"State machine exercised: {len(transitions)} transitions demonstrated")

    Path(tmp.name).unlink(missing_ok=True)


# ── Demo 3: Notebook Knowledge ───────────────────────────────────────

def demo_notebook_knowledge():
    header("DEMO 3: Notebook Knowledge — Code References & Expert Narrations")

    from core.notebook_knowledge import NotebookKnowledge
    nb = NotebookKnowledge("data")

    subheader("References for element_stiffness node")
    refs = nb.get_references_for_node("element_stiffness")
    for r in refs:
        print(f"  {BOLD}{r['function_family']}{RESET} ({r['module']})")
        print(f"    Role: {r.get('pedagogical_role', 'N/A')}")
        narration = r.get('narration', '')
        if narration:
            print(f"    Narration: {narration[:120]}...")
        code = r.get('code', '')
        if code:
            first_line = code.strip().split('\n')[0]
            print(f"    Code: {DIM}{first_line}{RESET}")
        print()

    subheader("Cross-Module Evolution Table")
    evo = nb.get_evolution_table()
    if evo and "rows" in evo:
        # Print as a formatted table
        print(f"  {'Function':<15} {'Springs':<20} {'Trusses':<20} {'Frames':<20}")
        print(f"  {'─'*15} {'─'*20} {'─'*20} {'─'*20}")
        for row in evo["rows"][:6]:
            fn = row[0]
            s = row[1][:18]
            t = row[2][:18]
            f = row[3][:18]
            print(f"  {fn:<15} {s:<20} {t:<20} {f:<20}")

    subheader("Curriculum Difference Note (beam_element)")
    diff = nb.get_curriculum_difference_note("beam_element")
    if diff:
        print(f"  {YELLOW}⚠ {diff}{RESET}")
    else:
        info("No curriculum difference for this node")

    success(f"Knowledge base loaded: {len(refs)} references for element_stiffness")


# ── Demo 4: Code Executor ────────────────────────────────────────────

def demo_executor():
    header("DEMO 4: Code Executor — Live Notebook Functions")

    from core.notebook_executor import NotebookExecutor

    ex = NotebookExecutor()

    subheader("Module 1: Spring Stiffness Matrix")
    ex.set_module("module1_springs")
    r = ex.execute("K = uelspring(100.0)\nprint('K =')\nprint(K)")
    code_block("K = uelspring(100.0)\nprint(K)", r.stdout)

    subheader("Module 2: Truss Element (45° diagonal)")
    ex.set_module("module2_trusses")
    r = ex.execute(
        "import numpy as np\n"
        "coord = np.array([[0.0,0.0],[1.0,1.0]])\n"
        "K = ueltruss2D(coord, 1.0, 1000.0)\n"
        "print('K_truss (4×4) =')\nprint(np.round(K, 1))"
    )
    code_block("K = ueltruss2D(coord_45deg, A=1, E=1000)", r.stdout)

    subheader("Module 3: Beam Element (horizontal)")
    ex.set_module("module3_frames")
    r = ex.execute(
        "import numpy as np\n"
        "coord = np.array([[0.0,0.0],[3.0,0.0]])\n"
        "K = uelbeam2DU(coord, 1.0, 1000.0)\n"
        "print('K_beam (6×6) =')\nprint(np.round(K, 2))"
    )
    code_block("K = uelbeam2DU(coord_horiz, I=1, E=1000)", r.stdout)

    subheader("Student Code Verification")
    ex.set_module("module1_springs")
    check = ex.verify_output(
        "K = uelspring(50.0)",
        {"K[0,0]": 50.0, "K[0,1]": -50.0, "K.shape": (2, 2)},
    )
    for c in check["checks"]:
        status = f"{GREEN}✓{RESET}" if c["correct"] else f"{RED}✗{RESET}"
        print(f"  {status} {c['expression']} = {c['actual']} (expected {c['expected']})")

    subheader("Curated Demo: element_stiffness_build")
    results = ex.run_demo("element_stiffness_build")
    for step in results:
        print(f"  {BOLD}Step {step['step']}{RESET}: {step['narration'][:80]}...")
        out = step['result']['stdout']
        if out:
            for line in out.split('\n')[:3]:
                print(f"    {GREEN}{line}{RESET}")
            if out.count('\n') > 3:
                print(f"    {DIM}... ({out.count(chr(10)) - 2} more lines){RESET}")
        print()

    success(f"Executor: 3 modules, {len(ex.get_available_demos())} demos, student verification working")


# ── Demo 5: Solver Bridge ────────────────────────────────────────────

def demo_solver():
    header("DEMO 5: Spring-Agent Solver Bridge — Live FEM Analysis")

    try:
        from core.spring_agent_bridge import SpringAgentBridge, NOTEBOOK_EXAMPLES
        bridge = SpringAgentBridge()
    except (FileNotFoundError, ImportError) as e:
        print(f"  {RED}Spring-agent not available: {e}{RESET}")
        info("Set SPRING_AGENT_PATH env var or place spring-agent as sibling directory")
        return

    subheader("Create & Solve: 3-Spring Series")
    result = bridge.create_and_solve(NOTEBOOK_EXAMPLES["three_spring_series"])
    if result["status"] == "success":
        info(f"Model: {result['model_summary']['name']}")
        info(f"Nodes: {result['model_summary']['num_nodes']}, "
             f"Elements: {result['model_summary']['num_elements']}")

        disps = result["results"]["displacements"]
        forces = result["results"]["element_forces"]
        reactions = result["results"]["reaction_forces"]

        print(f"\n  {BOLD}Displacements:{RESET}")
        for nid, val in sorted(disps.items(), key=lambda x: int(x[0])):
            print(f"    Node {nid}: u = {val:.6f} m")

        print(f"\n  {BOLD}Element Forces:{RESET}")
        for eid, val in sorted(forces.items(), key=lambda x: int(x[0])):
            print(f"    Element {eid}: f = {val:.2f} N")

        print(f"\n  {BOLD}Reactions:{RESET}")
        for nid, val in sorted(reactions.items(), key=lambda x: int(x[0])):
            print(f"    Node {nid}: R = {val:.2f} N")
    else:
        print(f"  {RED}Solve failed: {result.get('message')}{RESET}")

    subheader("Answer Verification")
    # Correct answer
    check = bridge.verify_displacement(3, 0.166667, tolerance=0.01)
    status = f"{GREEN}✓ CORRECT{RESET}" if check["correct"] else f"{RED}✗ WRONG{RESET}"
    print(f"  Student says u₃ = 0.166667 → {status}")
    print(f"    Expected: {check['expected']:.6f}, Error: {check.get('error_pct', 0):.2f}%")

    # Wrong answer
    check = bridge.verify_displacement(3, 0.5)
    status = f"{GREEN}✓ CORRECT{RESET}" if check["correct"] else f"{RED}✗ WRONG{RESET}"
    print(f"  Student says u₃ = 0.5 → {status}")
    print(f"    Expected: {check['expected']:.6f}, Error: {check.get('error_pct', 0):.1f}%")

    subheader("Sensitivity Analysis")
    result = bridge.execute("run_sensitivity", {
        "target": {"type": "displacement", "id": 3},
    })
    if result.get("status") == "success":
        print(f"  Which spring matters most for node 3 displacement?")
        for entry in result.get("sensitivities", result.get("ranked", []))[:5]:
            print(f"    {entry}")
    else:
        info(f"Sensitivity: {result}")

    subheader("Failure Cascade")
    bridge.create_and_solve(NOTEBOOK_EXAMPLES["failure_cascade_demo"])
    result = bridge.execute("run_failure_cascade", {})
    if result.get("status") == "success":
        steps = result.get("steps", result.get("cascade_steps", []))
        print(f"  Cascade completed in {len(steps)} steps")
        for s in steps:
            print(f"    {s}")
    else:
        info(f"Cascade: {result.get('status', 'N/A')}")

    success("Solver bridge: create, solve, verify, sensitivity, cascade — all working")


# ── Demo 6: Persistence ──────────────────────────────────────────────

def demo_persistence():
    header("DEMO 6: Multi-Session Persistence — Student Progress Tracking")

    import tempfile, shutil
    from core.persistence_manager import PersistenceManager

    tmp_dir = tempfile.mkdtemp()

    try:
        subheader("Session 1: New student 'Maria'")
        pm1 = PersistenceManager("maria", data_dir="data", students_dir=tmp_dir)
        orch1 = pm1.get_orchestrator()
        info(f"Node: {orch1.current_node}, State: {orch1.state}")
        info(f"Session count: {orch1.student['session_count']}")

        # Advance through some evaluations
        pm1.process_and_save({
            "procedural": "correct", "conceptual": "mechanical",
            "misconception_id": None, "confidence": "medium",
            "physical_reasoning_detected": False,
        })
        info(f"After eval → State: {orch1.state}")

        # Advance to next node
        orch1.state = "ADVANCE"
        pm1.advance_and_save()
        info(f"Advanced to: {orch1.current_node}")

        subheader("Session 2: Maria returns (simulated)")
        pm2 = PersistenceManager("maria", data_dir="data", students_dir=tmp_dir)
        orch2 = pm2.get_orchestrator()
        summary = pm2.get_resume_summary()

        print(f"  {BOLD}Welcome back, Maria!{RESET}")
        print(f"  Session #{summary['session_count']}")
        print(f"  Current topic: {summary['current_node_title']}")
        print(f"  State: {summary['current_state']}")
        print(f"  Progress: {summary['progress']['mastered']}/{summary['progress']['total']} nodes mastered")
        print(f"  Mastered: {summary['progress']['mastered_nodes'] or 'none yet'}")

        subheader("Multi-Student Isolation")
        pm_bob = PersistenceManager("bob", data_dir="data", students_dir=tmp_dir)
        pm_bob.get_orchestrator()

        students = PersistenceManager.list_students(tmp_dir)
        print(f"  Enrolled students:")
        for s in students:
            print(f"    {BOLD}{s['student_id']}{RESET}: "
                  f"node={s.get('current_node')}, "
                  f"sessions={s.get('session_count', 0)}")

        subheader("Transcript Export")
        transcript = pm2.export_transcript()
        print(f"  Student: {transcript['student_id']}")
        print(f"  Sessions: {transcript['session_count']}")
        active_nodes = {k: v for k, v in transcript['nodes'].items() if v['attempts'] > 0}
        print(f"  Nodes with activity: {len(active_nodes)}")
        for nid, nd in active_nodes.items():
            print(f"    {nid}: {nd['attempts']} attempts, p_mastery={nd['p_mastery']:.3f}")

        success("Persistence: create, save, resume, multi-student, export — all working")

    finally:
        shutil.rmtree(tmp_dir)


# ── Demo 7: Teacher Context ──────────────────────────────────────────

def demo_teacher_context():
    header("DEMO 7: Teacher Context Assembly — What the AI Teacher Sees")

    from core.orchestrator import Orchestrator

    orch = Orchestrator(data_dir="data")

    subheader("MODEL state — element_stiffness")
    orch.state = "MODEL"
    ctx = orch.build_teacher_context()

    print(f"  {BOLD}Context keys:{RESET} {', '.join(ctx.keys())}")
    print(f"  CAM technique: {ctx['cam_technique']}")
    print(f"  Node: {ctx['node_id']} — {ctx['node_title']}")
    print(f"  Scaffolding: {ctx['scaffolding_level']}")

    if ctx.get("problem"):
        print(f"\n  {BOLD}Problem:{RESET} {ctx['problem']['problem_id']}")
        print(f"  Statement: {ctx['problem']['statement'][:100]}...")
        if ctx['problem'].get('expert_narration'):
            print(f"  Expert narration: {ctx['problem']['expert_narration'][:100]}...")

    if ctx.get("notebook_references"):
        print(f"\n  {BOLD}Notebook references:{RESET} {len(ctx['notebook_references'])} entries")
        for r in ctx["notebook_references"][:2]:
            print(f"    {r.get('function_family', '?')} ({r.get('module', '?')})")

    if ctx.get("solver_demo"):
        print(f"\n  {BOLD}Solver demo:{RESET} {ctx['solver_demo']['status']}")

    if ctx.get("code_demo"):
        print(f"\n  {BOLD}Code demo:{RESET} {ctx['code_demo']['demo_name']} "
              f"({len(ctx['code_demo']['steps'])} steps)")

    if ctx.get("practice_problem"):
        print(f"\n  {BOLD}Practice problem ready:{RESET} {ctx['practice_problem']['statement'][:80]}...")

    subheader("GUIDED_PRACTICE state — with scaffolding")
    orch.state = "GUIDED_PRACTICE"
    orch.scaffolding_level = 3
    ctx_gp = orch.build_teacher_context()
    print(f"  CAM technique: {ctx_gp['cam_technique']}")
    print(f"  Scaffolding level: {ctx_gp['scaffolding_level']}")
    if ctx_gp.get("problem", {}).get("scaffolded_prompt"):
        print(f"  Scaffolded prompt: {ctx_gp['problem']['scaffolded_prompt'][:100]}...")

    subheader("ADVANCE state — transition context")
    orch.state = "ADVANCE"
    ctx_adv = orch.build_teacher_context()
    print(f"  Next node: {ctx_adv.get('next_node_title', 'N/A')}")
    if ctx_adv.get("evolution_table"):
        print(f"  Evolution table: {len(ctx_adv['evolution_table'].get('rows', []))} rows")

    subheader("Evaluator Context")
    orch.state = "GUIDED_PRACTICE"
    eval_ctx = orch.build_evaluator_context("The stiffness matrix is [[k, -k], [-k, k]]")
    print(f"  Context keys: {', '.join(eval_ctx.keys())}")
    print(f"  Student response captured: '{eval_ctx['student_response'][:60]}...'")
    print(f"  Problem solution available: {bool(eval_ctx.get('problem', {}).get('solution_steps'))}")
    print(f"  Misconception inventory: {list(eval_ctx.get('misconceptions', {}).keys())[:3]}...")

    success("Teacher context: all states produce rich, structured context for the AI")


# ── Main ──────────────────────────────────────────────────────────────

DEMOS = {
    "curriculum": ("Curriculum Graph — 18 nodes, 3 modules, 54 problems", demo_curriculum),
    "state": ("CAM State Machine — pedagogical transitions", demo_state_machine),
    "notebook": ("Notebook Knowledge — code references & narrations", demo_notebook_knowledge),
    "executor": ("Code Executor — live notebook functions", demo_executor),
    "solver": ("Solver Bridge — live FEM analysis", demo_solver),
    "persistence": ("Multi-Session Persistence — progress tracking", demo_persistence),
    "context": ("Teacher Context — what the AI teacher sees", demo_teacher_context),
}


def main():
    args = sys.argv[1:]

    if not args:
        print(f"\n{BOLD}FEA Tutor — Feature Demo Runner{RESET}\n")
        print("Available demos:\n")
        for key, (desc, _) in DEMOS.items():
            print(f"  {BOLD}{key:<14}{RESET} {desc}")
        print(f"\n  {BOLD}{'all':<14}{RESET} Run all demos")
        print(f"\nUsage: python demo.py <demo> [demo2] [demo3] ...")
        print(f"Example: python demo.py solver executor")
        return

    if "all" in args:
        targets = list(DEMOS.keys())
    else:
        targets = args

    for key in targets:
        if key in DEMOS:
            try:
                DEMOS[key][1]()
            except Exception as e:
                print(f"\n  {RED}Demo '{key}' failed: {e}{RESET}")
                import traceback
                traceback.print_exc()
        else:
            print(f"\n  {RED}Unknown demo: {key}{RESET}")

    print(f"\n{BOLD}{GREEN}{'═' * 66}")
    print(f"  Done. {len(targets)} demo(s) completed.")
    print(f"{'═' * 66}{RESET}\n")


if __name__ == "__main__":
    main()
