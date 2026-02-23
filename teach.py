#!/usr/bin/env python3
"""
FEA Tutor — Interactive Teaching Session

Jump directly to any node/state and have a real conversation with the
AI teacher, powered by all integrated knowledge layers.

Usage:
    python teach.py                                  # start from beginning
    python teach.py --node assembly                  # jump to assembly node
    python teach.py --node assembly --state MODEL    # jump to MODEL state
    python teach.py --node bar_element               # jump to Module 2
    python teach.py --student maria                   # resume Maria's session
    python teach.py --list-nodes                      # show all available nodes

Requires: ANTHROPIC_API_KEY environment variable (or .env file)
"""
from dotenv import load_dotenv
load_dotenv()

import os
import sys
import json
import argparse
import tempfile
import textwrap
from pathlib import Path

try:
    import anthropic
except ImportError:
    print("Error: pip install anthropic")
    sys.exit(1)


# ── Formatting ────────────────────────────────────────────────────────

BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
CYAN = "\033[96m"


# ── Teacher Prompt Builder ────────────────────────────────────────────

def build_teacher_system_prompt(context: dict) -> str:
    """Build the full system prompt for the teacher from orchestrator context."""

    cam_instructions = {
        "modeling": (
            "You are DEMONSTRATING this concept as an expert. Walk through the "
            "solution step-by-step, making your reasoning visible. Explain not just "
            "WHAT to do but WHY — connect every step to physical intuition. "
            "After the demonstration, present the practice problem and ask the "
            "student to try it."
        ),
        "coaching": (
            "The student is PRACTICING. Your role is to COACH, not solve. "
            "Provide hints calibrated to the scaffolding level. "
            "Level 3 = heavy guidance (almost give the answer). "
            "Level 2 = moderate (point to the right equation). "
            "Level 1 = light (ask leading questions). "
            "Level 0 = no scaffolding (let them work independently). "
            "When the student submits an answer, evaluate it and give feedback."
        ),
        "articulation": (
            "The student gave a correct answer but may not deeply understand WHY. "
            "PROBE their reasoning: ask them to explain their approach, predict "
            "what would change if a parameter changed, or connect the math to "
            "physical behavior. Do NOT accept 'I just followed the formula.'"
        ),
        "remediation": (
            "The student has a MISCONCEPTION. Address it directly but kindly. "
            "Show why their intuition is wrong using a concrete example or "
            "thought experiment, then guide them to the correct understanding."
        ),
        "exploration": (
            "This is an UNSCAFFOLDED ASSESSMENT. Present the problem and let "
            "the student work independently. Do not provide hints unless they "
            "are completely stuck. Evaluate their response rigorously."
        ),
        "assessment": (
            "ASSESS what the student already knows about this topic. Ask an "
            "open-ended question that reveals both procedural knowledge and "
            "conceptual understanding. Do not teach yet."
        ),
        "transition": (
            "The student has MASTERED this concept. Celebrate their achievement, "
            "summarize what they learned, and introduce the next topic. "
            "Explain how the new concept builds on what they just learned."
        ),
    }

    cam = context.get("cam_technique", "modeling")
    technique_instruction = cam_instructions.get(cam, cam_instructions["modeling"])

    parts = []

    # Core identity
    parts.append(
        "You are an expert Finite Element Analysis tutor teaching a graduate-level "
        "course on computational structural mechanics. You are warm, rigorous, and "
        "deeply committed to building physical intuition — not just procedural skill.\n"
    )

    # CAM technique
    parts.append(f"## Current Teaching Mode: {cam.upper()}\n{technique_instruction}\n")

    # Topic
    parts.append(
        f"## Topic: {context.get('node_title', 'Unknown')}\n"
        f"Learning objectives:\n"
    )
    for lo in context.get("learning_objectives", []):
        parts.append(f"- {lo}")
    parts.append("")

    # Problem
    problem = context.get("problem")
    if problem:
        parts.append(f"## Problem: {problem['problem_id']}")
        parts.append(f"Statement: {problem['statement']}\n")

        if cam == "modeling" and problem.get("expert_narration"):
            parts.append(f"## Expert Narration (use this to guide your demonstration):")
            parts.append(problem["expert_narration"])
            parts.append("")

        if problem.get("scaffolded_prompt") and cam == "coaching":
            parts.append(f"## Scaffolded Prompt (level {context.get('scaffolding_level', '?')}):")
            parts.append(problem["scaffolded_prompt"])
            parts.append("")

        if problem.get("solution_steps"):
            parts.append("## Solution (for your reference — do NOT reveal directly):")
            for i, step in enumerate(problem["solution_steps"]):
                parts.append(f"  Step {i+1}: {step}")
            parts.append(f"  Final answer: {problem.get('final_answer', 'N/A')}")
            parts.append("")

    # Practice problem (shown at end of MODEL)
    practice = context.get("practice_problem")
    if practice and cam == "modeling":
        parts.append("## Practice Problem (present this AFTER your demonstration):")
        parts.append(practice["statement"])
        if practice.get("scaffolded_prompt"):
            parts.append(f"Scaffolded hint: {practice['scaffolded_prompt']}")
        parts.append("")

    # Notebook references
    refs = context.get("notebook_references", [])
    if refs:
        parts.append("## Notebook Code References")
        parts.append("You have access to actual course notebook code. Reference it naturally:")
        for r in refs:
            parts.append(f"\n### {r.get('function_family', '?')} ({r.get('module', '?')})")
            if r.get("narration"):
                parts.append(r["narration"])
            if r.get("code"):
                parts.append(f"```python\n{r['code']}\n```")
        parts.append("")

    # Code demo results
    code_demo = context.get("code_demo")
    if code_demo:
        parts.append("## Live Code Demonstration Results")
        parts.append("You have actual execution output. Walk the student through each step:\n")
        for step in code_demo["steps"]:
            parts.append(f"**Step {step['step']}**: {step['narration']}")
            parts.append(f"```python\n{step['code']}\n```")
            parts.append(f"Output:\n```\n{step['result']['stdout']}\n```\n")

    # Solver demo results
    solver = context.get("solver_demo")
    if solver and solver.get("status") == "success":
        parts.append("## Live Solver Verification")
        parts.append("Use these computed values to verify hand calculations:\n")
        results = solver.get("results", {})
        if results.get("displacements"):
            parts.append(f"Displacements: {results['displacements']}")
        if results.get("element_forces"):
            parts.append(f"Element forces: {results['element_forces']}")
        if results.get("reaction_forces"):
            parts.append(f"Reactions: {results['reaction_forces']}")
        parts.append("")

    # Curriculum difference warning
    diff = context.get("curriculum_difference")
    if diff:
        parts.append(f"## ⚠ Curriculum Note\n{diff}\n")

    # Evolution table (for ADVANCE)
    evo = context.get("evolution_table")
    if evo and cam == "transition":
        parts.append("## Cross-Module Evolution")
        parts.append("Use this table to show how concepts scale across modules:")
        for row in evo.get("rows", []):
            parts.append(f"  {row}")
        parts.append("")

    # Next node (for ADVANCE)
    if context.get("next_node_title") and cam == "transition":
        parts.append(f"## Next Topic: {context['next_node_title']}")
        parts.append("Bridge from the current concept to the next one.\n")

    # Misconceptions to watch for
    misconceptions = context.get("misconceptions", {})
    if misconceptions:
        parts.append("## Misconceptions to Watch For")
        for mid, desc in misconceptions.items():
            parts.append(f"- {mid}: {desc}")
        parts.append("")

    # Student summary
    summary = context.get("student_summary", {})
    if summary:
        parts.append(
            f"## Student Status: p(mastery)={summary.get('p_mastery', 0):.2f}, "
            f"attempts={summary.get('attempts', 0)}"
        )
        unresolved = [m for m in summary.get("misconceptions_observed", [])
                      if m not in summary.get("misconceptions_resolved", [])]
        if unresolved:
            parts.append(f"Unresolved misconceptions: {unresolved}")
        parts.append("")

    # Welcome back context
    welcome = context.get("welcome_context")
    if welcome and welcome.get("is_returning_student"):
        rs = welcome["resume_summary"]
        parts.append(
            f"## Returning Student (session #{rs['session_count']})\n"
            f"Welcome them back. They were working on {rs['current_node_title']}."
        )
        if welcome.get("recap_needed"):
            parts.append(f"Recap: {welcome['recap_hint']}")
        parts.append("")

    return "\n".join(parts)


# ── Evaluator Prompt Builder ──────────────────────────────────────────

EVALUATOR_SYSTEM = textwrap.dedent("""\
    You are a precise evaluator for a Finite Element Analysis course.
    Analyze the student's response and classify it.

    Return a JSON object with exactly these keys:
    {
        "procedural": "correct" | "minor_error" | "major_error" | "incomplete",
        "conceptual": "deep" | "mechanical" | "absent",
        "misconception_id": null or "M1" or "M2" or "M3" etc,
        "confidence": "high" | "medium" | "low",
        "physical_reasoning_detected": true | false,
        "brief_feedback": "one sentence summary for the teacher"
    }

    Definitions:
    - procedural: Is the math/procedure correct?
    - conceptual: Does the student show physical understanding (deep),
      follow steps without understanding (mechanical), or show no reasoning (absent)?
    - misconception_id: If a known misconception is detected, return its ID.
    - physical_reasoning_detected: Did the student connect math to physics?

    Return ONLY the JSON object, no other text.
""")


def build_evaluator_prompt(eval_context: dict) -> str:
    """Build the evaluator user prompt from orchestrator context."""
    parts = [f"Student response: \"{eval_context['student_response']}\"\n"]

    parts.append(f"Topic: {eval_context.get('node_id', '?')}")
    parts.append(f"Learning objectives: {eval_context.get('learning_objectives', [])}\n")

    problem = eval_context.get("problem", {})
    if problem:
        parts.append(f"Problem: {problem.get('problem_id', '?')}")
        parts.append(f"Expected solution steps: {problem.get('solution_steps', [])}")
        parts.append(f"Final answer: {problem.get('final_answer', 'N/A')}")
        if problem.get("evaluation_markers"):
            parts.append(f"Evaluation markers: {problem['evaluation_markers']}")

    misconceptions = eval_context.get("misconceptions", {})
    if misconceptions:
        parts.append(f"\nKnown misconceptions for this topic:")
        for mid, desc in misconceptions.items():
            parts.append(f"  {mid}: {desc}")

    markers = eval_context.get("physical_reasoning_markers", [])
    if markers:
        parts.append(f"\nPhysical reasoning markers to look for: {markers}")

    return "\n".join(parts)


# ── Interactive Session ───────────────────────────────────────────────

class TeachingSession:
    """Interactive teaching session with Claude as teacher and evaluator."""

    def __init__(self, orchestrator, persistence_mgr=None, model="claude-sonnet-4-5-20250929"):
        self.orch = orchestrator
        self.pm = persistence_mgr
        self.client = anthropic.Anthropic()
        self.model = model
        self.conversation_history = []

    def run(self):
        """Main interaction loop."""
        print(f"\n{BOLD}{CYAN}╔══════════════════════════════════════════════════════╗")
        print(f"║          FEA Tutor — Interactive Session             ║")
        print(f"╚══════════════════════════════════════════════════════╝{RESET}\n")

        print(f"  Node:  {BOLD}{self.orch.current_node}{RESET} — {self.orch.node_data['title']}")
        print(f"  State: {BOLD}{self.orch.state}{RESET}")
        print(f"  Scaffolding: {self.orch.scaffolding_level}")
        print(f"\n  Commands: {DIM}/skip  /repeat  /state  /node <name>  /quit{RESET}\n")

        # Teacher always speaks first — present problem, ask question, etc.
        self._teacher_turn()

        while True:
            try:
                user_input = input(f"\n{GREEN}Student:{RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nSession ended.")
                break

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                if self._handle_command(user_input):
                    continue
                else:
                    break  # /quit

            # Student message → evaluate if needed, then teacher responds
            self._process_student_turn(user_input)

    def _teacher_turn(self, extra_context: dict = None):
        """Generate and display a teacher response."""
        context = self.orch.build_teacher_context()
        if extra_context:
            context.update(extra_context)

        system_prompt = build_teacher_system_prompt(context)

        # Build messages
        messages = list(self.conversation_history)
        if not messages:
            # First turn — teacher initiates
            messages.append({
                "role": "user",
                "content": "Please begin the lesson.",
            })

        print(f"\n{CYAN}Teacher:{RESET} ", end="", flush=True)

        # Stream the response
        full_response = ""
        with self.client.messages.stream(
            model=self.model,
            max_tokens=2000,
            system=system_prompt,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
                full_response += text

        print()  # newline after stream

        # Track conversation
        if not self.conversation_history:
            self.conversation_history.append({
                "role": "user",
                "content": "Please begin the lesson.",
            })
        self.conversation_history.append({
            "role": "assistant",
            "content": full_response,
        })

    def _process_student_turn(self, student_message: str):
        """Process a student message: evaluate → transition → teacher responds."""
        # Add to conversation
        self.conversation_history.append({
            "role": "user",
            "content": student_message,
        })

        if self.orch.needs_evaluation():
            # Call evaluator
            eval_context = self.orch.build_evaluator_context(student_message)
            evaluation = self._evaluate(eval_context)

            if evaluation:
                print(f"\n  {DIM}[Eval: procedural={evaluation['procedural']}, "
                      f"conceptual={evaluation['conceptual']}, "
                      f"misconception={evaluation.get('misconception_id', 'none')}]{RESET}")

                old_state = self.orch.state
                if self.pm:
                    self.pm.process_and_save(evaluation)
                else:
                    self.orch.process_evaluation(evaluation)

                new_state = self.orch.state
                if old_state != new_state:
                    print(f"  {DIM}[Transition: {old_state} → {new_state}, "
                          f"scaffolding={self.orch.scaffolding_level}]{RESET}")

                    # Handle ADVANCE
                    if new_state == "ADVANCE":
                        self._teacher_turn()
                        print(f"\n  {DIM}[Advancing to next node...]{RESET}")
                        if self.pm:
                            self.pm.advance_and_save()
                        else:
                            self.orch.advance_to_next_node()
                        print(f"  {DIM}[Now at: {self.orch.current_node} — "
                              f"{self.orch.node_data['title']}]{RESET}")

                # Teacher responds with updated context
                self._teacher_turn({"evaluator_feedback": evaluation.get("brief_feedback", "")})
            else:
                # Evaluation failed — teacher responds without evaluation
                self._teacher_turn()
        else:
            # No evaluation needed (e.g., MODEL state) — teacher just responds
            self._teacher_turn()

    def _evaluate(self, eval_context: dict) -> dict | None:
        """Call Claude as evaluator and parse the response."""
        prompt = build_evaluator_prompt(eval_context)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                system=EVALUATOR_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )

            text = response.content[0].text.strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                text = text.rsplit("```", 1)[0]

            return json.loads(text)
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            print(f"  {RED}[Evaluator parse error: {e}]{RESET}")
            return None
        except Exception as e:
            print(f"  {RED}[Evaluator error: {e}]{RESET}")
            return None

    def _handle_command(self, cmd: str) -> bool:
        """Handle a slash command. Returns True to continue, False to quit."""
        parts = cmd.split()
        command = parts[0].lower()

        if command == "/quit":
            print("Session ended.")
            return False

        elif command == "/skip":
            self.orch.handle_student_override("skip")
            if self.pm:
                self.pm.save()
            print(f"  {DIM}[Skipped to {self.orch.state}]{RESET}")
            self._teacher_turn()

        elif command == "/repeat":
            self.orch.handle_student_override("repeat")
            if self.pm:
                self.pm.save()
            print(f"  {DIM}[Returning to MODEL]{RESET}")
            self.conversation_history = []
            self._teacher_turn()

        elif command == "/state":
            print(f"\n  {self.orch.get_status_summary()}")

        elif command == "/node" and len(parts) > 1:
            node_name = parts[1]
            if node_name in self.orch.curriculum["nodes"]:
                self.orch.student["current_node"] = node_name
                self.orch.state = "MODEL"
                self.orch.scaffolding_level = 3
                if self.pm:
                    self.pm.save()
                self.conversation_history = []
                print(f"  {DIM}[Jumped to {node_name} — {self.orch.node_data['title']}]{RESET}")
                self._teacher_turn()
            else:
                print(f"  {RED}Unknown node: {node_name}{RESET}")
                print(f"  Available: {', '.join(sorted(self.orch.curriculum['nodes'].keys()))}")

        elif command == "/help":
            print(f"""
  {BOLD}Commands:{RESET}
    /skip       — Skip to mastery assessment
    /repeat     — Re-watch the MODEL demonstration
    /state      — Show current status (node, mastery, scaffolding)
    /node NAME  — Jump to a specific curriculum node
    /quit       — End session
    /help       — Show this help
""")
        else:
            print(f"  {DIM}Unknown command: {cmd}. Type /help{RESET}")

        return True


# ── Main ──────────────────────────────────────────────────────────────

def list_nodes():
    """Print all available nodes."""
    from core.orchestrator import Orchestrator
    orch = Orchestrator(data_dir="data")

    edges = {s: t for s, t in orch.curriculum["edges"]}
    node = "element_stiffness"
    modules = [
        ("Module 1 — Springs", 6),
        ("Module 2 — Trusses", 6),
        ("Module 3 — Frames", 6),
    ]
    idx = 0
    for mod_name, count in modules:
        print(f"\n{BOLD}{mod_name}{RESET}")
        for i in range(count):
            if node is None:
                break
            nd = orch.curriculum["nodes"][node]
            print(f"  {node:<30} {nd['title']}")
            node = edges.get(node)
        idx += count


def main():
    parser = argparse.ArgumentParser(
        description="FEA Tutor — Interactive Teaching Session"
    )
    parser.add_argument("--node", type=str, help="Jump to a specific curriculum node")
    parser.add_argument("--state", type=str, default=None,
                        help="Start in a specific state (MODEL, GUIDED_PRACTICE, etc.)")
    parser.add_argument("--student", type=str, default=None,
                        help="Student ID for persistent sessions")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-5-20250929",
                        help="Claude model to use")
    parser.add_argument("--list-nodes", action="store_true",
                        help="List all available curriculum nodes")
    args = parser.parse_args()

    if args.list_nodes:
        list_nodes()
        return

    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(f"{RED}Error: Set ANTHROPIC_API_KEY environment variable{RESET}")
        sys.exit(1)

    # Set up orchestrator
    pm = None
    if args.student:
        from core.persistence_manager import PersistenceManager
        pm = PersistenceManager(student_id=args.student)
        orch = pm.get_orchestrator()

        if orch.student.get("session_count", 0) > 1:
            summary = pm.get_resume_summary()
            print(f"\n{BOLD}Welcome back!{RESET} Session #{summary['session_count']}")
            print(f"Last topic: {summary['current_node_title']}")
            print(f"Progress: {summary['progress']['mastered']}/{summary['progress']['total']} nodes")
    else:
        from core.orchestrator import Orchestrator
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.close()
        os.unlink(tmp.name)
        orch = Orchestrator(data_dir="data", student_file=tmp.name)

    # Jump to specific node if requested
    if args.node:
        if args.node in orch.curriculum["nodes"]:
            orch.student["current_node"] = args.node
            orch.state = args.state or "MODEL"
            orch.scaffolding_level = 3
            if pm:
                pm.save()
        else:
            print(f"{RED}Unknown node: {args.node}{RESET}")
            list_nodes()
            sys.exit(1)
    elif args.state:
        orch.state = args.state
        if pm:
            pm.save()

    # Start session
    session = TeachingSession(orch, persistence_mgr=pm, model=args.model)
    session.run()


if __name__ == "__main__":
    main()
