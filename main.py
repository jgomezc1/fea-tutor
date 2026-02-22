"""
FEA Tutor — Main Application Loop

This is the entry point for the tutoring system. It ties together:
- The Orchestrator (pedagogical state machine)
- The Teacher Agent (student-facing LLM)
- The Evaluator Agent (analytical classification LLM)

Usage:
    python main.py                    # New session
    python main.py --resume state.json  # Resume from saved state
    python main.py --student-id alice   # Named student
"""
from dotenv import load_dotenv
load_dotenv()
import argparse
import json
import sys
from pathlib import Path

import anthropic

from core.orchestrator import Orchestrator
from agents.teacher import teach
from agents.evaluator import evaluate


# ANSI colors for terminal UI
class Colors:
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def print_status(orchestrator: Orchestrator):
    """Print the current status bar."""
    print(f"\n{Colors.DIM}─── {orchestrator.get_status_summary()} ───{Colors.RESET}\n")


def print_teacher(text: str):
    """Print teacher output."""
    print(f"\n{Colors.GREEN}{Colors.BOLD}Professor FEA:{Colors.RESET}")
    print(f"{Colors.GREEN}{text}{Colors.RESET}")


def print_evaluation_debug(evaluation: dict):
    """Print evaluation details for debugging."""
    print(f"\n{Colors.DIM}─── Evaluator Output ───")
    print(f"  Procedural:  {evaluation.get('procedural', 'N/A')}")
    print(f"  Conceptual:  {evaluation.get('conceptual', 'N/A')}")
    print(f"  Physical reasoning: {evaluation.get('physical_reasoning_detected', 'N/A')}")
    print(f"  Misconception: {evaluation.get('misconception_id', 'None')}")
    print(f"  Confidence:  {evaluation.get('confidence', 'N/A')}")
    print(f"  Evidence:    {evaluation.get('evidence', 'N/A')[:100]}...")
    print(f"  Suggestion:  {evaluation.get('suggested_action', 'N/A')}")
    print(f"───────────────────────{Colors.RESET}\n")


def run_session(orchestrator: Orchestrator, debug: bool = False):
    """
    Main interaction loop.

    The loop follows this sequence each turn:
    1. Student sends message
    2. If evaluation needed: call evaluator, update student model, transition state
    3. Build teacher context for (possibly new) state
    4. Call teacher agent
    5. Display response
    6. Save student model
    """
    client = anthropic.Anthropic()
    conversation_history = []

    print(f"\n{Colors.BOLD}{'='*60}")
    print("  FEA Tutor — Direct Stiffness Method")
    print(f"  Cognitive Apprenticeship Model (CAM)")
    print(f"{'='*60}{Colors.RESET}")
    print(f"\n{Colors.CYAN}Type your responses. Commands: /skip /repeat /status /quit{Colors.RESET}\n")

    # --- Initial teacher message (ASSESS_PRIOR opening) ---
    if debug:
        print_status(orchestrator)

    context = orchestrator.build_teacher_context()
    try:
        response = teach(context, "", [], client)
    except anthropic.APIError as e:
        print(f"\n{Colors.RED}API unavailable ({e}). Session state saved. "
              f"Resume with: python main.py --resume {orchestrator.student_file}{Colors.RESET}")
        orchestrator.save_student_model()
        return
    print_teacher(response)
    conversation_history.append({"role": "assistant", "content": response})

    # MODEL is a single-exchange demonstration: transition to GUIDED_PRACTICE
    # so the student's first response is evaluated as a practice attempt
    if orchestrator.state == "MODEL":
        orchestrator.state = "GUIDED_PRACTICE"
        print(f"{Colors.YELLOW}[TRACE] MODEL → GUIDED_PRACTICE (auto-transition after demonstration){Colors.RESET}")

    # --- Main loop ---
    while True:
        # Get student input
        try:
            student_input = input(f"\n{Colors.BLUE}{Colors.BOLD}You: {Colors.RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{Colors.YELLOW}Session ended.{Colors.RESET}")
            break

        if not student_input:
            continue

        # Handle commands
        if student_input.startswith("/"):
            cmd = student_input.lower()
            if cmd == "/quit":
                orchestrator.save_student_model()
                print(f"\n{Colors.YELLOW}Session saved. Goodbye!{Colors.RESET}")
                break
            elif cmd == "/status":
                print_status(orchestrator)
                continue
            elif cmd == "/skip":
                orchestrator.handle_student_override("skip")
                print(f"{Colors.CYAN}Skipping to mastery assessment...{Colors.RESET}")
            elif cmd == "/repeat":
                orchestrator.handle_student_override("repeat")
                print(f"{Colors.CYAN}Returning to modeling...{Colors.RESET}")
            elif cmd == "/debug":
                debug = not debug
                print(f"{Colors.CYAN}Debug mode: {'ON' if debug else 'OFF'}{Colors.RESET}")
                continue
            else:
                print(f"{Colors.YELLOW}Unknown command. Available: /skip /repeat /status /quit /debug{Colors.RESET}")
                continue

        # Track pre-evaluation state to detect mastery-assessment transitions
        pre_eval_state = orchestrator.state

        try:
            # Step 2: Evaluate if needed
            if orchestrator.needs_evaluation() and not student_input.startswith("/"):
                eval_context = orchestrator.build_evaluator_context(student_input)
                evaluation = evaluate(eval_context, client)

                if debug:
                    print_evaluation_debug(evaluation)

                # Update student model and determine state transition
                old_state = orchestrator.state
                p_mastery_before = orchestrator.student_node["p_mastery"]
                orchestrator.process_evaluation(evaluation)
                new_state = orchestrator.state
                p_mastery_after = orchestrator.student_node["p_mastery"]

                # ── TRACE: state transition details ──
                print(f"\n{Colors.YELLOW}[TRACE] process_evaluation complete:{Colors.RESET}")
                print(f"{Colors.YELLOW}[TRACE]   state: {old_state} → {new_state}{Colors.RESET}")
                print(f"{Colors.YELLOW}[TRACE]   p_mastery: {p_mastery_before:.4f} → {p_mastery_after:.4f} (threshold=0.90){Colors.RESET}")
                print(f"{Colors.YELLOW}[TRACE]   level0_passed: {orchestrator.student_node.get('level0_passed', False)}{Colors.RESET}")
                unresolved = [m for m in orchestrator.student_node["misconceptions_observed"]
                              if m not in orchestrator.student_node["misconceptions_resolved"]]
                print(f"{Colors.YELLOW}[TRACE]   unresolved_misconceptions: {unresolved}{Colors.RESET}")
                print(f"{Colors.YELLOW}[TRACE]   scaffolding_level: {orchestrator.scaffolding_level}{Colors.RESET}")

            # ── TRACE: check which branch we take ──
            print(f"\n{Colors.YELLOW}[TRACE] Branch check: orchestrator.state = '{orchestrator.state}', pre_eval_state = '{pre_eval_state}'{Colors.RESET}")

            # ── Branch 1: ADVANCE — mastery confirmed, bridge to next node ──
            if orchestrator.state == "ADVANCE":
                print(f"{Colors.YELLOW}[TRACE] >>> ENTERING ADVANCE BLOCK <<<{Colors.RESET}")

                if debug:
                    print_status(orchestrator)

                context = orchestrator.build_teacher_context()

                print(f"{Colors.YELLOW}[TRACE]   cam_technique: {context['cam_technique']}{Colors.RESET}")
                print(f"{Colors.YELLOW}[TRACE]   node_title: {context['node_title']}{Colors.RESET}")
                print(f"{Colors.YELLOW}[TRACE]   next_node_title: {context.get('next_node_title', '(MISSING)')}{Colors.RESET}")

                transition_prompt = (
                    "[SYSTEM: The student has just passed the mastery assessment for "
                    + context["node_title"] + ". Generate your transition response as "
                    "instructed: congratulate them, summarize key insights, bridge to "
                    + context.get("next_node_title", "the next topic") + ", and begin "
                    "assessing their prior knowledge of it. "
                    "Do NOT end the session or say farewell — the session is continuing.]"
                )

                print(f"{Colors.YELLOW}[TRACE]   Calling teach() with conversation history + transition directive{Colors.RESET}")
                response = teach(context, transition_prompt, conversation_history, client)

                conversation_history.append({"role": "user", "content": student_input})
                conversation_history.append({"role": "assistant", "content": response})

                if len(conversation_history) > 40:
                    conversation_history = conversation_history[-40:]

                print_teacher(response)

                has_next = orchestrator.advance_to_next_node()
                print(f"{Colors.YELLOW}[TRACE]   advance_to_next_node() returned: {has_next}{Colors.RESET}")
                print(f"{Colors.YELLOW}[TRACE]   new node: {orchestrator.current_node}, new state: {orchestrator.state}{Colors.RESET}")
                orchestrator.student["interaction_count"] += 1
                orchestrator.save_student_model()

                if not has_next:
                    break

                if debug:
                    print_status(orchestrator)

                continue

            # ── Branch 2: Near-mastery — student gave a strong ASSESS_MASTERY answer
            #    but check_mastery failed (BKT < 0.90 or unresolved misconceptions).
            #    Without this branch, the teacher gets a "coaching" prompt with the
            #    full conversation history showing a mastery-level answer, and the
            #    LLM interprets it as a natural conclusion → generates a farewell. ──
            if pre_eval_state == "ASSESS_MASTERY" and orchestrator.state != "ADVANCE":
                print(f"{Colors.YELLOW}[TRACE] >>> ENTERING NEAR-MASTERY BLOCK (was ASSESS_MASTERY, now {orchestrator.state}) <<<{Colors.RESET}")

                if debug:
                    print_status(orchestrator)

                context = orchestrator.build_teacher_context()

                print(f"{Colors.YELLOW}[TRACE]   cam_technique: {context['cam_technique']}{Colors.RESET}")
                print(f"{Colors.YELLOW}[TRACE]   Calling teach() with EMPTY history + near-mastery directive{Colors.RESET}")

                near_mastery_prompt = (
                    "[SYSTEM: The student just responded to a mastery assessment for "
                    + context["node_title"] + ". They showed promising understanding but "
                    "need more practice before we can confirm full mastery. "
                    "Acknowledge specifically what they did well, then present the next "
                    "practice problem for THIS SAME TOPIC (" + context["node_title"] + "). "
                    "Be encouraging — they are close.\n"
                    "CRITICAL CONSTRAINTS:\n"
                    "- Do NOT mention the next topic or transitioning to a new concept.\n"
                    "- Do NOT say goodbye, farewell, or end the session.\n"
                    "- Do NOT use bridging language like 'moving on' or 'next we will cover'.\n"
                    "- Stay focused on the CURRENT topic. We are continuing practice here.]"
                )
                response = teach(context, near_mastery_prompt, [], client)

                conversation_history.append({"role": "user", "content": student_input})
                conversation_history.append({"role": "assistant", "content": response})

                if len(conversation_history) > 40:
                    conversation_history = conversation_history[-40:]

                print_teacher(response)

                orchestrator.student["interaction_count"] += 1
                orchestrator.save_student_model()
                continue

            # ── Branch 3: Regular teacher call (all other states) ──
            if debug:
                print_status(orchestrator)

            context = orchestrator.build_teacher_context()
            print(f"{Colors.YELLOW}[TRACE]   Regular teach() call: cam_technique={context['cam_technique']}, node={context['node_title']}, history_len={len(conversation_history)}{Colors.RESET}")
            response = teach(context, student_input, conversation_history, client)

            # Update conversation history
            conversation_history.append({"role": "user", "content": student_input})
            conversation_history.append({"role": "assistant", "content": response})

            # Keep conversation history manageable (last 20 turns)
            if len(conversation_history) > 40:
                conversation_history = conversation_history[-40:]

            print_teacher(response)

            # MODEL is a single-exchange demonstration: transition to
            # GUIDED_PRACTICE so the next student response is evaluated
            if orchestrator.state == "MODEL":
                orchestrator.state = "GUIDED_PRACTICE"
                print(f"{Colors.YELLOW}[TRACE] MODEL → GUIDED_PRACTICE (auto-transition after demonstration){Colors.RESET}")

            # Step 6: Save
            orchestrator.student["interaction_count"] += 1
            orchestrator.save_student_model()

        except anthropic.APIError as e:
            orchestrator.save_student_model()
            print(f"\n{Colors.RED}API unavailable after retries ({e}).{Colors.RESET}")
            print(f"{Colors.RED}Your progress has been saved. You can:{Colors.RESET}")
            print(f"{Colors.RED}  - Type another message to retry{Colors.RESET}")
            print(f"{Colors.RED}  - /quit to exit (resume later with --resume){Colors.RESET}")


def main():
    parser = argparse.ArgumentParser(description="FEA Tutor — Cognitive Apprenticeship Model")
    parser.add_argument("--resume", type=str, help="Path to saved student state JSON")
    parser.add_argument("--student-id", type=str, default="default", help="Student identifier")
    parser.add_argument("--debug", action="store_true", help="Show evaluation and state details")
    args = parser.parse_args()

    # Determine student file path
    if args.resume:
        student_file = args.resume
    else:
        student_file = f"data/student_{args.student_id}.json"

    # Initialize orchestrator
    orchestrator = Orchestrator(
        data_dir="data",
        student_file=student_file if Path(student_file).exists() else None,
    )

    if not Path(student_file).exists():
        orchestrator.student["student_id"] = args.student_id
        orchestrator.student_file = student_file

    print(f"{Colors.DIM}Student: {args.student_id} | "
          f"State file: {student_file} | "
          f"Debug: {args.debug}{Colors.RESET}")

    run_session(orchestrator, debug=args.debug)


if __name__ == "__main__":
    main()
