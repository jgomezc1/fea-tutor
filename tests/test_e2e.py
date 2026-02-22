"""
End-to-end simulation — calls the actual LLM evaluator with scripted student
personas to validate the full Node 1 (element_stiffness) cycle.

Three personas:
  (a) Strong student  — deep physical reasoning, should reach ADVANCE quickly
  (b) Mechanical student — correct but pattern-based, longer path through DIAGNOSE
  (c) Struggling student — errors and misconceptions, needs remediation

Run:  pytest tests/test_e2e.py -v -s   (use -s to see the full report)

Requires: ANTHROPIC_API_KEY environment variable set.
"""

import pytest
import time
from dotenv import load_dotenv
load_dotenv()

import anthropic

from core.orchestrator import Orchestrator
from agents.evaluator import evaluate


# ---------------------------------------------------------------------------
# Scripted student responses — keyed by problem_id (or "assess_prior")
# ---------------------------------------------------------------------------

STRONG_STUDENT = {
    "assess_prior": (
        "For a spring connecting node 1 to node 2, I'd start from equilibrium "
        "at each node. The spring extension is delta = u2 - u1. If the spring "
        "is in tension, it pulls node 1 toward node 2, so the external force "
        "at node 1 must oppose that: F1 = k*u1 - k*u2. By Newton's third law, "
        "F2 = -k*u1 + k*u2. In matrix form that gives [[k, -k], [-k, k]]. "
        "Each column sums to zero because a free spring must satisfy overall "
        "equilibrium — there's no net external force on an unsupported element."
    ),
    "ES_A": (
        "Let me derive this from scratch using equilibrium. The spring extension "
        "is delta = u2 - u1. The internal spring force is f = k*delta = k(u2-u1). "
        "At node 1, if the spring is in tension (u2 > u1), it pulls node 1 to "
        "the right, toward node 2. The external force needed at node 1 to maintain "
        "equilibrium is F1 = -f = k*u1 - k*u2. "
        "At node 2, the spring pulls it to the left (toward node 1), so "
        "F2 = f = -k*u1 + k*u2. "
        "Assembling: [F1, F2]^T = [[k, -k], [-k, k]] [u1, u2]^T "
        "= [[200, -200], [-200, 200]] [u1, u2]^T. "
        "Verification: each column sums to zero, which is equilibrium of the "
        "free body — no net force on the unsupported spring. Diagonal terms are "
        "positive because the spring resists displacement at each node."
    ),
    "ES_B": (
        "(a) The first column represents the forces needed when u1=1, u2=0. "
        "Physically: I push node 1 one unit right while holding node 2 fixed. "
        "The spring compresses. At node 1, the spring pushes back (restoring "
        "force) with magnitude k. At node 2, the compressed spring pushes it "
        "left with force -k. So the column is [k, -k]^T — these are the "
        "external forces I need to maintain that displacement state. "
        "(b) Diagonal terms are positive because the spring always resists the "
        "displacement of its own node — it's a restoring force. Off-diagonal "
        "terms are negative because displacing one node transmits force to the "
        "other node in the opposite direction through the spring coupling. "
        "(c) Each column sums to zero: k + (-k) = 0. This is Newton's second "
        "law for the free spring — if the system has no supports, the net "
        "external force must be zero. A non-zero column sum would mean the "
        "spring as a whole is accelerating, which is unphysical."
    ),
    "ES_C": (
        "(a) Setting up Ku = F: k*u1 - k*u2 = 100 and -k*u1 + k*u2 = -100. "
        "Both equations say k(u1-u2) = 100, so u1 - u2 = 0.5. But I can't "
        "find u1 and u2 individually. Physically this makes sense: the spring "
        "is floating freely. I know how much it stretches (the relative "
        "displacement) but not where it is in space. Any rigid body translation "
        "is a valid solution: u1=0, u2=-0.5 works, and so does u1=5, u2=4.5. "
        "(b) Now F1=100, F2=50. Adding the equations: 0 = 150, a contradiction. "
        "This loading violates free body equilibrium — the net force is 150 N "
        "but there's nothing to resist it. The spring would accelerate forever. "
        "The matrix correctly rejects this unphysical scenario. "
        "(c) The matrix is singular because det = k^2 - k^2 = 0. The null "
        "space is [1,1]^T — both nodes moving equally with no deformation. "
        "This is the rigid body mode. The singularity is not an error; it's "
        "the matrix correctly representing that an unsupported element can "
        "translate freely. Boundary conditions will remove the rigid body mode "
        "and make the system solvable."
    ),
}

MECHANICAL_STUDENT = {
    "assess_prior": (
        "The stiffness matrix for a spring element is [[k, -k], [-k, k]]. "
        "I remember from the textbook that the diagonal entries are k and "
        "the off-diagonal entries are -k. For k=200, it would be "
        "[[200, -200], [-200, 200]]."
    ),
    "ES_A": (
        "The element stiffness matrix relates forces to displacements via F = Ku. "
        "For a spring element, the pattern is: positive k on the diagonal and "
        "negative k on the off-diagonal. So K = [[k, -k], [-k, k]]. "
        "With k = 200 N/m, that gives K = [[200, -200], [-200, 200]]. "
        "I know this because the stiffness matrix always has this symmetric "
        "pattern for a 1D spring element."
    ),
    "ES_B": (
        "(a) The first column is [k, -k]^T, which is just the first column "
        "of the matrix. "
        "(b) The diagonal is positive because k is positive. The off-diagonal "
        "is negative because that's the pattern — positive on diagonal, "
        "negative off-diagonal. It's always like that for springs. "
        "(c) Each column sums to zero: k + (-k) = 0. The entries cancel out."
    ),
    "ES_C": (
        "(a) I set up Ku = F and try to solve. I get two equations that are "
        "dependent, so there's no unique solution. I think this means the "
        "matrix doesn't work properly for this case. "
        "(b) For the second case, the equations give 0 = 150 which is a "
        "contradiction, so there's no solution. "
        "(c) The determinant is zero so the matrix is singular. I think we "
        "need to add boundary conditions to fix the matrix so it can be "
        "inverted."
    ),
}

STRUGGLING_STUDENT = {
    "assess_prior": (
        "I'm not really sure. I know F = kx for a spring. So maybe the "
        "stiffness matrix is just [[k, 0], [0, k]]? Each node has force "
        "k times its displacement."
    ),
    "ES_A": (
        "OK so F = Ku. At node 1, F1 = k*u1. At node 2, F2 = k*u2. "
        "So the matrix is [[k, 0], [0, k]] = [[200, 0], [0, 200]]. "
        "Each node just has F = kx applied to it independently."
    ),
    "ES_B": (
        "I'm not sure what the columns mean physically. The first column "
        "is [k, -k] I guess. The diagonal is positive and off-diagonal is "
        "negative. I don't really know why the off-diagonal has to be "
        "negative though. For part (c), the sum is zero but I'm not sure "
        "what that means."
    ),
    "ES_C": (
        "(a) I try to solve but I can't get unique values for u1 and u2. "
        "I think there's no solution. "
        "(b) This also has no solution since 0 = 150 is impossible. "
        "(c) The matrix must be defective since we can't solve either "
        "problem. Maybe it's wrong and needs to be fixed."
    ),
}

PERSONAS = {
    "strong": STRONG_STUDENT,
    "mechanical": MECHANICAL_STUDENT,
    "struggling": STRUGGLING_STUDENT,
}


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------

def run_persona(name: str, responses: dict, client: anthropic.Anthropic,
                max_turns: int = 20) -> dict:
    """
    Run a persona through the Node 1 (element_stiffness) CAM cycle.

    Returns a report dict with transitions, evaluations, final state, etc.
    """
    orch = Orchestrator(data_dir="data")
    orch.student_file = f"/tmp/test_e2e_{name}.json"

    report = {
        "persona": name,
        "turns": [],
        "final_state": None,
        "final_node": None,
        "final_p_mastery": None,
        "advance_reached": False,
    }

    for turn in range(1, max_turns + 1):
        state_before = orch.state
        node = orch.current_node

        # Stop if we've advanced past element_stiffness
        if node != "element_stiffness":
            break

        # MODEL auto-transition (no evaluation, no student response)
        if orch.state == "MODEL":
            orch.state = "GUIDED_PRACTICE"
            report["turns"].append({
                "turn": turn,
                "state_before": "MODEL",
                "state_after": "GUIDED_PRACTICE",
                "action": "auto-transition",
                "evaluation": None,
            })
            continue

        # States that need evaluation
        if orch.state in ["GUIDED_PRACTICE", "DIAGNOSE", "ASSESS_MASTERY",
                          "ASSESS_PRIOR", "REMEDIATE"]:
            # Select the appropriate response
            problem = orch.get_current_problem()
            if problem:
                problem_id = problem["problem_id"]
                response_text = responses.get(problem_id, responses["assess_prior"])
            else:
                response_text = responses["assess_prior"]

            # Call actual LLM evaluator
            eval_context = orch.build_evaluator_context(response_text)
            evaluation = evaluate(eval_context, client)

            # Process and record
            orch.process_evaluation(evaluation)
            state_after = orch.state

            report["turns"].append({
                "turn": turn,
                "state_before": state_before,
                "state_after": state_after,
                "problem_id": problem["problem_id"] if problem else None,
                "action": "evaluation",
                "evaluation": {
                    "procedural": evaluation.get("procedural"),
                    "conceptual": evaluation.get("conceptual"),
                    "physical_reasoning": evaluation.get("physical_reasoning_detected"),
                    "confidence": evaluation.get("confidence"),
                    "misconception_id": evaluation.get("misconception_id"),
                    "suggested_action": evaluation.get("suggested_action"),
                },
            })

            # Check for ADVANCE
            if orch.state == "ADVANCE":
                report["advance_reached"] = True
                orch.advance_to_next_node()
                break

            continue

        # Unexpected state — break to avoid infinite loop
        report["turns"].append({
            "turn": turn,
            "state_before": state_before,
            "state_after": orch.state,
            "action": f"unhandled state: {orch.state}",
            "evaluation": None,
        })
        break

    sn = orch.student["nodes"]["element_stiffness"]
    report["final_state"] = orch.state
    report["final_node"] = orch.current_node
    report["final_p_mastery"] = sn["p_mastery"]
    report["level0_passed"] = sn["level0_passed"]
    report["misconceptions_observed"] = sn["misconceptions_observed"]
    report["misconceptions_resolved"] = sn["misconceptions_resolved"]
    report["total_attempts"] = sn["attempts"]

    return report


def print_report(report: dict):
    """Pretty-print a persona simulation report."""
    name = report["persona"].upper()
    print(f"\n{'='*70}")
    print(f"  PERSONA: {name}")
    print(f"{'='*70}")

    for t in report["turns"]:
        e = t.get("evaluation")
        if t["action"] == "auto-transition":
            print(f"  Turn {t['turn']:2d}: MODEL → GUIDED_PRACTICE  (auto-transition)")
        elif e:
            prob = t.get("problem_id", "—")
            proc = e["procedural"]
            conc = e["conceptual"]
            phys = "phys" if e["physical_reasoning"] else "no-phys"
            conf = e["confidence"]
            misc = e["misconception_id"] or "—"
            print(f"  Turn {t['turn']:2d}: {t['state_before']:17s} → {t['state_after']:17s}  "
                  f"[{prob}]  {proc}/{conc}/{phys}/{conf}  misc={misc}")
        else:
            print(f"  Turn {t['turn']:2d}: {t['state_before']:17s} → {t['state_after']:17s}  "
                  f"({t['action']})")

    print(f"\n  Result:")
    print(f"    ADVANCE reached:    {report['advance_reached']}")
    print(f"    Final node:         {report['final_node']}")
    print(f"    Final state:        {report['final_state']}")
    print(f"    Final p_mastery:    {report['final_p_mastery']:.4f}")
    print(f"    Level-0 passed:     {report['level0_passed']}")
    print(f"    Total attempts:     {report['total_attempts']}")
    print(f"    Misconceptions:     observed={report['misconceptions_observed']}  "
          f"resolved={report['misconceptions_resolved']}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Helper: detect state machine loops
# ---------------------------------------------------------------------------

def detect_loops(report: dict) -> list[str]:
    """Detect repeating state transition patterns in a report."""
    loops = []
    turns = report["turns"]
    for i in range(2, len(turns)):
        # Check for 2-state oscillation: A→B→A→B
        if (i >= 3
                and turns[i].get("state_before") == turns[i-2].get("state_before")
                and turns[i].get("state_after") == turns[i-2].get("state_after")
                and turns[i-1].get("state_before") == turns[i-3].get("state_before") if i >= 3 else False):
            pair = f"{turns[i]['state_before']} ↔ {turns[i]['state_after']}"
            if pair not in loops:
                loops.append(pair)
    return loops


# ---------------------------------------------------------------------------
# Pytest tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def api_client():
    """Shared Anthropic client for all e2e tests."""
    return anthropic.Anthropic()


class TestStrongStudent:
    def test_reaches_advance(self, api_client):
        """Strong student with deep physical reasoning should reach ADVANCE."""
        report = run_persona("strong", STRONG_STUDENT, api_client)
        print_report(report)

        assert report["advance_reached"], (
            f"Strong student did not reach ADVANCE after {len(report['turns'])} turns. "
            f"Final state: {report['final_state']}, p_mastery: {report['final_p_mastery']:.4f}"
        )
        assert report["final_node"] == "local_global_dofs"
        assert report["final_p_mastery"] > 0.90

        # Verify evaluator classified responses as deep
        deep_count = sum(
            1 for t in report["turns"]
            if (t.get("evaluation") or {}).get("conceptual") == "deep"
        )
        assert deep_count >= 2, f"Expected at least 2 deep classifications, got {deep_count}"

        # Should be fast — BKT crosses 0.90 after 2 correct observations
        assert len(report["turns"]) <= 6, (
            f"Strong student took {len(report['turns'])} turns — expected ≤6"
        )


class TestMechanicalStudent:
    def test_triggers_diagnose_loop(self, api_client):
        """Mechanical student should trigger DIAGNOSE repeatedly.

        Known behavior: the mechanical student gets stuck in a
        GUIDED_PRACTICE ↔ DIAGNOSE loop because their scripted responses
        are always classified as 'mechanical'. This is the correct
        evaluator classification — the orchestrator's lack of an
        escalation path is a documented design limitation (CLAUDE.md:
        'Common Pitfalls > State machine loops').
        """
        report = run_persona("mechanical", MECHANICAL_STUDENT, api_client,
                             max_turns=12)
        print_report(report)

        # Evaluator should consistently classify as mechanical
        mechanical_evals = sum(
            1 for t in report["turns"]
            if (t.get("evaluation") or {}).get("conceptual") == "mechanical"
        )
        print(f"  [check] Mechanical evaluations: {mechanical_evals}")
        assert mechanical_evals >= 3, (
            f"Expected at least 3 mechanical classifications, got {mechanical_evals}"
        )

        # Should have hit DIAGNOSE (articulation probe)
        diagnose_count = sum(
            1 for t in report["turns"]
            if t.get("state_after") == "DIAGNOSE"
        )
        print(f"  [check] DIAGNOSE transitions: {diagnose_count}")
        assert diagnose_count >= 1, "Mechanical student should trigger DIAGNOSE"

        # Should NOT advance — mechanical responses never satisfy mastery
        assert not report["advance_reached"], (
            "Mechanical student should not advance with pattern-based answers"
        )

        # Flag detected loops for visibility
        loops = detect_loops(report)
        if loops:
            print(f"  [known issue] State machine loops detected: {loops}")


class TestStrugglingStudent:
    def test_triggers_errors_and_remediation(self, api_client):
        """Struggling student should trigger errors, misconceptions, and
        remediation cycles.

        Known behavior: the struggling student's responses consistently
        trigger misconception detection, creating a GUIDED_PRACTICE ↔
        REMEDIATE loop. This is the correct evaluator behavior — the
        scripted responses genuinely contain misconceptions (e.g., M3:
        'believing stiffness depends on forces').
        """
        report = run_persona("struggling", STRUGGLING_STUDENT, api_client,
                             max_turns=12)
        print_report(report)

        # Should have errors detected
        error_evals = sum(
            1 for t in report["turns"]
            if (t.get("evaluation") or {}).get("procedural") in ["major_error", "incomplete"]
        )
        print(f"  [check] Error evaluations: {error_evals}")
        assert error_evals >= 1, "Struggling student should have at least one error"

        # Should NOT advance
        assert not report["advance_reached"], (
            "Struggling student should not advance with misconceived answers"
        )

        # p_mastery should remain low
        assert report["final_p_mastery"] < 0.50, (
            f"Struggling student p_mastery unexpectedly high: "
            f"{report['final_p_mastery']:.4f}"
        )

        # Should have misconceptions observed
        print(f"  [check] Misconceptions observed: {report['misconceptions_observed']}")
        assert len(report["misconceptions_observed"]) >= 1, (
            "Evaluator should detect at least one misconception"
        )

        # Flag detected loops
        loops = detect_loops(report)
        if loops:
            print(f"  [known issue] State machine loops detected: {loops}")
