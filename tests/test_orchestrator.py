"""
Unit tests for the Orchestrator state machine.

All tests use mock evaluator outputs — no LLM calls.
Run: pytest tests/test_orchestrator.py -v
"""

import pytest
import json
import copy
from pathlib import Path

from core.orchestrator import Orchestrator, PROBLEM_MAP
from core.bkt import update_mastery, check_mastery


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def orch():
    """Fresh orchestrator starting at element_stiffness / ASSESS_PRIOR."""
    o = Orchestrator(data_dir="data")
    # Override student_file to avoid writing to disk during tests
    o.student_file = "/tmp/test_student_state.json"
    return o


# ---------------------------------------------------------------------------
# Evaluation factories — shorthand for common evaluator outputs
# ---------------------------------------------------------------------------

def eval_deep(physical=True, confidence="high", misconception=None):
    return {
        "procedural": "correct",
        "conceptual": "deep",
        "misconception_id": misconception,
        "physical_reasoning_detected": physical,
        "confidence": confidence,
        "suggested_action": "advance",
    }


def eval_mechanical():
    return {
        "procedural": "correct",
        "conceptual": "mechanical",
        "misconception_id": None,
        "physical_reasoning_detected": False,
        "confidence": "medium",
        "suggested_action": "articulation_probe",
    }


def eval_major_error(misconception=None):
    return {
        "procedural": "major_error",
        "conceptual": "absent",
        "misconception_id": misconception,
        "physical_reasoning_detected": False,
        "confidence": "medium",
        "suggested_action": "re_model",
    }


def eval_minor_error_deep():
    return {
        "procedural": "minor_error",
        "conceptual": "deep",
        "misconception_id": None,
        "physical_reasoning_detected": True,
        "confidence": "medium",
        "suggested_action": "practice",
    }


def eval_incomplete():
    return {
        "procedural": "incomplete",
        "conceptual": "absent",
        "misconception_id": None,
        "physical_reasoning_detected": False,
        "confidence": "low",
        "suggested_action": "re_model",
    }


def eval_absent_reasoning():
    return {
        "procedural": "correct",
        "conceptual": "absent",
        "misconception_id": None,
        "physical_reasoning_detected": False,
        "confidence": "low",
        "suggested_action": "articulation_probe",
    }


def eval_confused(misconception=None):
    """Major error with some reasoning shown — hits the scaffolding-increase
    branch in GP (unlike absent which routes to DIAGNOSE)."""
    return {
        "procedural": "major_error",
        "conceptual": "mechanical",
        "misconception_id": misconception,
        "physical_reasoning_detected": False,
        "confidence": "medium",
        "suggested_action": "re_model",
    }


# ===========================================================================
# ASSESS_PRIOR transitions
# ===========================================================================

class TestAssessPrior:
    def test_deep_to_assess_mastery(self, orch):
        """correct + deep → ASSESS_MASTERY, scaffolding reset to 0."""
        assert orch.state == "ASSESS_PRIOR"
        orch.process_evaluation(eval_deep())
        assert orch.state == "ASSESS_MASTERY"
        assert orch.scaffolding_level == 0

    def test_mechanical_to_model(self, orch):
        """correct + mechanical → MODEL."""
        orch.process_evaluation(eval_mechanical())
        assert orch.state == "MODEL"

    def test_incomplete_to_model(self, orch):
        """incomplete → MODEL."""
        orch.process_evaluation(eval_incomplete())
        assert orch.state == "MODEL"

    def test_major_error_to_model(self, orch):
        """major_error → MODEL."""
        orch.process_evaluation(eval_major_error())
        assert orch.state == "MODEL"


# ===========================================================================
# MODEL auto-transition
# ===========================================================================

class TestModel:
    def test_model_not_in_needs_evaluation(self, orch):
        """MODEL is not an evaluated state — it's a single demonstration."""
        orch.state = "MODEL"
        assert not orch.needs_evaluation()

    def test_model_context_has_practice_problem(self, orch):
        """MODEL context includes the upcoming GUIDED_PRACTICE problem."""
        orch.state = "MODEL"
        ctx = orch.build_teacher_context()
        assert ctx["cam_technique"] == "modeling"
        assert "practice_problem" in ctx
        assert ctx["practice_problem"]["statement"]
        assert ctx["practice_problem"]["scaffolded_prompt"]

    def test_model_context_has_expert_narration(self, orch):
        """MODEL context includes expert narration for the demonstration."""
        orch.state = "MODEL"
        ctx = orch.build_teacher_context()
        assert "problem" in ctx
        assert "expert_narration" in ctx["problem"]
        assert len(ctx["problem"]["expert_narration"]) > 100

    def test_auto_transition_to_guided_practice(self, orch):
        """After MODEL demonstration, setting state to GP enables evaluation."""
        orch.state = "MODEL"
        assert not orch.needs_evaluation()
        # Simulate the auto-transition that main.py performs
        orch.state = "GUIDED_PRACTICE"
        assert orch.needs_evaluation()


# ===========================================================================
# GUIDED_PRACTICE — scaffolding fading
# ===========================================================================

class TestScaffoldingFade:
    def test_fade_from_3_to_2(self, orch):
        """correct + deep at scaffolding 3 → scaffolding 2, stay in GP."""
        orch.state = "GUIDED_PRACTICE"
        orch.scaffolding_level = 3
        orch.process_evaluation(eval_deep())
        assert orch.state == "GUIDED_PRACTICE"
        assert orch.scaffolding_level == 2

    def test_fade_from_2_to_1(self, orch):
        """correct + deep at scaffolding 2 → scaffolding 1, stay in GP."""
        orch.state = "GUIDED_PRACTICE"
        orch.scaffolding_level = 2
        orch.process_evaluation(eval_deep())
        assert orch.state == "GUIDED_PRACTICE"
        assert orch.scaffolding_level == 1

    def test_fade_from_1_to_assess_mastery(self, orch):
        """correct + deep at scaffolding 1 → ASSESS_MASTERY, scaffolding 0."""
        orch.state = "GUIDED_PRACTICE"
        orch.scaffolding_level = 1
        orch.process_evaluation(eval_deep())
        assert orch.state == "ASSESS_MASTERY"
        assert orch.scaffolding_level == 0

    def test_fade_from_0_to_assess_mastery(self, orch):
        """correct + deep at scaffolding 0 → ASSESS_MASTERY."""
        orch.state = "GUIDED_PRACTICE"
        orch.scaffolding_level = 0
        orch.process_evaluation(eval_deep())
        assert orch.state == "ASSESS_MASTERY"


# ===========================================================================
# GUIDED_PRACTICE — scaffolding increase on failure
# ===========================================================================

class TestScaffoldingIncrease:
    def test_major_error_increases_scaffolding(self, orch):
        """major_error (with some reasoning) at scaffolding 1 → scaffolding 2.
        Note: major_error + absent routes to DIAGNOSE instead; this tests the
        else branch which requires non-absent conceptual."""
        orch.state = "GUIDED_PRACTICE"
        orch.scaffolding_level = 1
        orch.process_evaluation(eval_confused())
        assert orch.state == "GUIDED_PRACTICE"
        assert orch.scaffolding_level == 2

    def test_major_error_at_max_scaffolding_to_model(self, orch):
        """major_error at scaffolding 3 → MODEL (can't increase further)."""
        orch.state = "GUIDED_PRACTICE"
        orch.scaffolding_level = 3
        orch.process_evaluation(eval_confused())
        assert orch.state == "MODEL"
        assert orch.scaffolding_level == 3

    def test_major_error_absent_increases_scaffolding(self, orch):
        """major_error + absent reasoning → increase scaffolding (student clearly lost)."""
        orch.state = "GUIDED_PRACTICE"
        orch.scaffolding_level = 2
        orch.process_evaluation(eval_major_error())
        assert orch.state == "GUIDED_PRACTICE"
        assert orch.scaffolding_level == 3

    def test_minor_error_deep_stays_in_practice(self, orch):
        """minor_error + deep reasoning → stay in GP at same level."""
        orch.state = "GUIDED_PRACTICE"
        orch.scaffolding_level = 2
        orch.process_evaluation(eval_minor_error_deep())
        assert orch.state == "GUIDED_PRACTICE"
        assert orch.scaffolding_level == 2


# ===========================================================================
# GUIDED_PRACTICE — misconception and DIAGNOSE triggers
# ===========================================================================

class TestGuidedPracticeRouting:
    def test_misconception_to_remediate(self, orch):
        """Misconception detected in GP → REMEDIATE (regardless of correctness)."""
        orch.state = "GUIDED_PRACTICE"
        orch.process_evaluation(eval_deep(misconception="M2"))
        assert orch.state == "REMEDIATE"

    def test_mechanical_to_diagnose(self, orch):
        """correct + mechanical → DIAGNOSE (need articulation)."""
        orch.state = "GUIDED_PRACTICE"
        orch.process_evaluation(eval_mechanical())
        assert orch.state == "DIAGNOSE"

    def test_absent_reasoning_to_diagnose(self, orch):
        """correct + absent reasoning → DIAGNOSE."""
        orch.state = "GUIDED_PRACTICE"
        orch.process_evaluation(eval_absent_reasoning())
        assert orch.state == "DIAGNOSE"


# ===========================================================================
# DIAGNOSE transitions
# ===========================================================================

class TestDiagnose:
    def test_deep_physical_low_scaffold_to_assess_mastery(self, orch):
        """deep + physical in DIAGNOSE with scaffolding <= 1 → ASSESS_MASTERY."""
        orch.state = "DIAGNOSE"
        orch.scaffolding_level = 1
        orch.process_evaluation(eval_deep(physical=True))
        assert orch.state == "ASSESS_MASTERY"
        assert orch.scaffolding_level == 0

    def test_deep_physical_high_scaffold_to_practice(self, orch):
        """deep + physical in DIAGNOSE with scaffolding > 1 → GP, scaffold fades."""
        orch.state = "DIAGNOSE"
        orch.scaffolding_level = 3
        orch.process_evaluation(eval_deep(physical=True))
        assert orch.state == "GUIDED_PRACTICE"
        assert orch.scaffolding_level == 2

    def test_misconception_to_remediate(self, orch):
        """Misconception in DIAGNOSE → REMEDIATE."""
        orch.state = "DIAGNOSE"
        orch.process_evaluation(eval_major_error(misconception="M1"))
        assert orch.state == "REMEDIATE"

    def test_mechanical_to_practice(self, orch):
        """Still mechanical in DIAGNOSE → back to GP."""
        orch.state = "DIAGNOSE"
        orch.process_evaluation(eval_mechanical())
        assert orch.state == "GUIDED_PRACTICE"


# ===========================================================================
# REMEDIATE transitions
# ===========================================================================

class TestRemediate:
    def test_remediate_always_to_practice(self, orch):
        """After remediation evaluation, always → GUIDED_PRACTICE."""
        orch.state = "REMEDIATE"
        # Even a perfect response after remediation goes to practice
        orch.process_evaluation(eval_deep())
        assert orch.state == "GUIDED_PRACTICE"

    def test_remediate_after_error_to_practice(self, orch):
        orch.state = "REMEDIATE"
        orch.process_evaluation(eval_major_error())
        assert orch.state == "GUIDED_PRACTICE"


# ===========================================================================
# ASSESS_MASTERY transitions
# ===========================================================================

class TestAssessMastery:
    def test_advance_normal_path(self, orch):
        """correct+deep+physical+high with BKT >= 0.90 and level0 → ADVANCE."""
        orch.state = "ASSESS_MASTERY"
        sn = orch.student_node
        # Set preconditions for mastery
        sn["p_mastery"] = 0.95
        sn["level0_passed"] = True
        orch.process_evaluation(eval_deep())
        # After eval, p_mastery goes even higher; check_mastery should pass
        assert sn["level0_passed"] is True
        assert orch.state == "ADVANCE"

    def test_advance_skip_override(self, orch):
        """'/skip' + perfect assessment → ADVANCE even with low BKT."""
        sn = orch.student_node
        assert sn["p_mastery"] == 0.1  # fresh student
        orch.handle_student_override("skip")
        assert orch.state == "ASSESS_MASTERY"
        assert orch._skipped_to_mastery is True
        orch.process_evaluation(eval_deep())
        assert orch.state == "ADVANCE"
        # BKT should be floored to threshold
        assert sn["p_mastery"] >= 0.90

    def test_skip_override_blocked_by_unresolved_misconception(self, orch):
        """'/skip' + perfect but unresolved misconception → NOT ADVANCE."""
        sn = orch.student_node
        sn["misconceptions_observed"].append("M1")
        orch.handle_student_override("skip")
        orch.process_evaluation(eval_deep())
        assert orch.state != "ADVANCE"
        # Should fall to check_mastery which also fails (BKT too low)
        assert orch.state == "GUIDED_PRACTICE"

    def test_near_mastery_bkt_below_threshold(self, orch):
        """Perfect assessment but BKT < 0.90 → GUIDED_PRACTICE (near-mastery)."""
        orch.state = "ASSESS_MASTERY"
        sn = orch.student_node
        sn["p_mastery"] = 0.1  # too low
        orch.process_evaluation(eval_deep())
        # level0_passed set to True, but BKT won't reach 0.90 from 0.1
        assert sn["level0_passed"] is True
        assert orch.state == "GUIDED_PRACTICE"
        assert orch.scaffolding_level == 1

    def test_fail_to_practice_with_scaffolding(self, orch):
        """Failed mastery assessment → GUIDED_PRACTICE with scaffolding 2."""
        orch.state = "ASSESS_MASTERY"
        orch.process_evaluation(eval_major_error())
        assert orch.state == "GUIDED_PRACTICE"
        assert orch.scaffolding_level == 2

    def test_misconception_to_remediate(self, orch):
        """Misconception in mastery assessment → REMEDIATE."""
        orch.state = "ASSESS_MASTERY"
        e = eval_major_error(misconception="M2")
        orch.process_evaluation(e)
        assert orch.state == "REMEDIATE"

    def test_skip_flag_cleared_after_use(self, orch):
        """The _skipped_to_mastery flag is consumed after one evaluation."""
        orch.handle_student_override("skip")
        assert orch._skipped_to_mastery is True
        # Fail the assessment
        orch.process_evaluation(eval_major_error())
        # Flag should be cleared
        assert orch._skipped_to_mastery is False


# ===========================================================================
# ADVANCE → next node
# ===========================================================================

class TestAdvance:
    def test_advance_to_local_global_dofs(self, orch):
        """advance_to_next_node → local_global_dofs, ASSESS_PRIOR, scaffold 3."""
        orch.state = "ADVANCE"
        result = orch.advance_to_next_node()
        assert result is True
        assert orch.current_node == "local_global_dofs"
        assert orch.state == "ASSESS_PRIOR"
        assert orch.scaffolding_level == 3

    def test_advance_context_includes_next_node(self, orch):
        """ADVANCE context includes next_node_title for bridging."""
        orch.state = "ADVANCE"
        ctx = orch.build_teacher_context()
        assert ctx["cam_technique"] == "transition"
        assert ctx["next_node_title"] == "Local vs. Global DOFs"


# ===========================================================================
# Misconception resolution
# ===========================================================================

class TestMisconceptionResolution:
    def test_resolved_on_targeted_problem(self, orch):
        """Deep physical reasoning on targeted problem resolves misconception."""
        sn = orch.student_node
        sn["misconceptions_observed"].append("M2")
        # ES_C targets M2; used in ASSESS_MASTERY for element_stiffness
        orch.state = "ASSESS_MASTERY"
        orch.process_evaluation(eval_deep())
        assert "M2" in sn["misconceptions_resolved"]

    def test_not_resolved_when_still_flagged(self, orch):
        """Misconception not resolved if evaluator still detects it."""
        sn = orch.student_node
        sn["misconceptions_observed"].append("M2")
        orch.state = "ASSESS_MASTERY"
        orch.process_evaluation(eval_deep(misconception="M2"))
        assert "M2" not in sn["misconceptions_resolved"]

    def test_not_resolved_on_untargeted_problem(self, orch):
        """Misconception not resolved on problem that doesn't target it."""
        sn = orch.student_node
        sn["misconceptions_observed"].append("M2")
        # ES_A targets no misconceptions; used in GUIDED_PRACTICE
        orch.state = "GUIDED_PRACTICE"
        orch.scaffolding_level = 1
        orch.process_evaluation(eval_deep())
        assert "M2" not in sn["misconceptions_resolved"]

    def test_not_resolved_when_mechanical(self, orch):
        """Mechanical response doesn't resolve misconceptions."""
        sn = orch.student_node
        sn["misconceptions_observed"].append("M2")
        orch.state = "ASSESS_MASTERY"
        orch.process_evaluation(eval_mechanical())
        assert "M2" not in sn["misconceptions_resolved"]

    def test_resolution_unblocks_advance(self, orch):
        """Resolving last misconception allows check_mastery to pass."""
        sn = orch.student_node
        sn["misconceptions_observed"].append("M2")
        sn["p_mastery"] = 0.95
        sn["level0_passed"] = True
        unresolved = [m for m in sn["misconceptions_observed"]
                      if m not in sn["misconceptions_resolved"]]
        assert not check_mastery(sn["p_mastery"], 0.90, True, unresolved)

        sn["misconceptions_resolved"].append("M2")
        unresolved = [m for m in sn["misconceptions_observed"]
                      if m not in sn["misconceptions_resolved"]]
        assert check_mastery(sn["p_mastery"], 0.90, True, unresolved)


# ===========================================================================
# LO status updates
# ===========================================================================

class TestLOStatus:
    def test_lo_demonstrated_on_deep(self, orch):
        """correct + deep marks targeted LOs as demonstrated."""
        orch.state = "GUIDED_PRACTICE"
        orch.scaffolding_level = 1
        sn = orch.student_node
        # GP at attempt 0 uses ES_A which targets LO1
        orch.process_evaluation(eval_deep())
        assert sn["lo_status"]["LO1"] == "demonstrated"

    def test_lo_mechanical_on_mechanical(self, orch):
        """correct + mechanical marks LOs as attempted_mechanical."""
        orch.state = "GUIDED_PRACTICE"
        sn = orch.student_node
        orch.process_evaluation(eval_mechanical())
        assert sn["lo_status"]["LO1"] == "attempted_mechanical"

    def test_lo_failed_on_major_error(self, orch):
        """major_error marks LOs as attempted_failed."""
        orch.state = "GUIDED_PRACTICE"
        sn = orch.student_node
        orch.process_evaluation(eval_major_error())
        assert sn["lo_status"]["LO1"] == "attempted_failed"


# ===========================================================================
# BKT mastery tracking
# ===========================================================================

class TestBKT:
    def test_mastery_increases_on_correct(self, orch):
        """p_mastery increases after a correct observation."""
        sn = orch.student_node
        initial = sn["p_mastery"]
        orch.state = "GUIDED_PRACTICE"
        orch.process_evaluation(eval_deep())
        assert sn["p_mastery"] > initial

    def test_mastery_decreases_on_incorrect(self, orch):
        """p_mastery changes after an incorrect observation."""
        sn = orch.student_node
        sn["p_mastery"] = 0.5
        orch.state = "GUIDED_PRACTICE"
        orch.scaffolding_level = 1
        orch.process_evaluation(eval_major_error())
        # With p_transit, mastery might not decrease below starting point
        # but it should be less than what a correct answer would give
        assert sn["p_mastery"] < 0.5  # incorrect pushes it down from 0.5

    def test_two_correct_crosses_threshold(self, orch):
        """Two consecutive correct observations push BKT above 0.90."""
        sn = orch.student_node
        params = orch.student["bkt_params"]
        p = sn["p_mastery"]  # 0.1
        p = update_mastery(p, True, params)   # ~0.733
        p = update_mastery(p, True, params)   # ~0.984
        assert p >= 0.90

    def test_attempts_incremented(self, orch):
        """Each process_evaluation increments attempts."""
        sn = orch.student_node
        assert sn["attempts"] == 0
        orch.state = "GUIDED_PRACTICE"
        orch.process_evaluation(eval_deep())
        assert sn["attempts"] == 1
        orch.process_evaluation(eval_mechanical())
        assert sn["attempts"] == 2


# ===========================================================================
# Student overrides
# ===========================================================================

class TestStudentOverrides:
    def test_skip_sets_assess_mastery(self, orch):
        orch.handle_student_override("skip")
        assert orch.state == "ASSESS_MASTERY"
        assert orch.scaffolding_level == 0
        assert orch._skipped_to_mastery is True

    def test_repeat_sets_model(self, orch):
        orch.state = "GUIDED_PRACTICE"
        orch.handle_student_override("repeat")
        assert orch.state == "MODEL"

    def test_ask_question_no_change(self, orch):
        original = orch.state
        orch.handle_student_override("ask_question")
        assert orch.state == original


# ===========================================================================
# Full Node 1 → Node 2 progression
# ===========================================================================

class TestFullProgression:
    def test_node1_to_node2_strong_student(self, orch):
        """Simulate a strong student progressing from element_stiffness to
        local_global_dofs through the full CAM cycle.

        Path: ASSESS_PRIOR → (deep, BKT 0.1→0.73) → ASSESS_MASTERY
              → (deep, BKT 0.73→0.98) → ADVANCE
              → local_global_dofs / ASSESS_PRIOR

        BKT crosses 0.90 after just 2 correct observations (0.1→0.73→0.98),
        so the strong student advances in 2 turns.
        """
        transitions = []
        sn = orch.student_node

        # Turn 1: ASSESS_PRIOR — student shows deep understanding
        assert orch.state == "ASSESS_PRIOR"
        orch.process_evaluation(eval_deep())
        transitions.append(("ASSESS_PRIOR", orch.state))
        assert orch.state == "ASSESS_MASTERY"
        assert sn["p_mastery"] > 0.70  # ~0.733 after first correct

        # Turn 2: ASSESS_MASTERY (ES_C) — deep, BKT jumps above 0.90
        orch.process_evaluation(eval_deep())
        transitions.append(("ASSESS_MASTERY", orch.state))
        assert sn["level0_passed"] is True
        assert sn["p_mastery"] > 0.90  # ~0.984 after second correct
        assert orch.state == "ADVANCE"

        # Advance to Node 2
        result = orch.advance_to_next_node()
        assert result is True
        assert orch.current_node == "local_global_dofs"
        assert orch.state == "ASSESS_PRIOR"
        assert orch.scaffolding_level == 3

        # Verify element_stiffness is mastered
        es_node = orch.student["nodes"]["element_stiffness"]
        assert es_node["p_mastery"] > 0.90
        assert es_node["level0_passed"] is True
        assert len(transitions) == 2

    def test_node1_mechanical_student_longer_path(self, orch):
        """A mechanical student takes more turns: ASSESS_PRIOR → MODEL →
        GP → DIAGNOSE → GP → ... eventually reaches mastery."""
        transitions = []

        # Turn 1: ASSESS_PRIOR — mechanical
        orch.process_evaluation(eval_mechanical())
        transitions.append(("ASSESS_PRIOR", orch.state))
        assert orch.state == "MODEL"

        # Simulate MODEL auto-transition
        orch.state = "GUIDED_PRACTICE"

        # Turn 2: GP at scaffolding 3 — mechanical again → DIAGNOSE
        orch.process_evaluation(eval_mechanical())
        transitions.append(("GUIDED_PRACTICE", orch.state))
        assert orch.state == "DIAGNOSE"

        # Turn 3: DIAGNOSE — still mechanical → back to GP
        orch.process_evaluation(eval_mechanical())
        transitions.append(("DIAGNOSE", orch.state))
        assert orch.state == "GUIDED_PRACTICE"

        # Turn 4: GP — now shows deep understanding, scaffold 3→2
        orch.process_evaluation(eval_deep())
        transitions.append(("GUIDED_PRACTICE", orch.state))
        assert orch.state == "GUIDED_PRACTICE"
        assert orch.scaffolding_level == 2

        # Turn 5: GP — deep again, scaffold 2→1
        orch.process_evaluation(eval_deep())
        transitions.append(("GUIDED_PRACTICE", orch.state))
        assert orch.state == "GUIDED_PRACTICE"
        assert orch.scaffolding_level == 1

        # Turn 6: GP — deep, scaffold 1→0, → ASSESS_MASTERY
        orch.process_evaluation(eval_deep())
        transitions.append(("GUIDED_PRACTICE", orch.state))
        assert orch.state == "ASSESS_MASTERY"

        # Turn 7: ASSESS_MASTERY — deep + high confidence
        # BKT after 4 correct + 2 "correct"(mechanical): should be high
        orch.process_evaluation(eval_deep())
        transitions.append(("ASSESS_MASTERY", orch.state))
        assert orch.state == "ADVANCE"

        assert len(transitions) == 7

    def test_struggling_student_with_misconception(self, orch):
        """A struggling student hits a misconception, gets remediated,
        and eventually recovers."""
        sn = orch.student_node

        # Turn 1: ASSESS_PRIOR — major error (absent goes to MODEL)
        orch.process_evaluation(eval_major_error())
        assert orch.state == "MODEL"

        # MODEL auto-transition
        orch.state = "GUIDED_PRACTICE"

        # Turn 2: GP — confused at scaffolding 3 → MODEL (can't increase)
        orch.scaffolding_level = 3
        orch.process_evaluation(eval_confused())
        assert orch.state == "MODEL"

        # MODEL auto-transition again
        orch.state = "GUIDED_PRACTICE"

        # Turn 3: GP — misconception detected → REMEDIATE
        orch.process_evaluation(eval_confused(misconception="M2"))
        assert orch.state == "REMEDIATE"
        assert "M2" in sn["misconceptions_observed"]

        # Turn 4: REMEDIATE — deep response on ES_C (targets M2) → GP
        # Since REMEDIATE maps to ES_C which targets M2, and the eval is
        # deep+correct+physical with no misconception flagged, M2 resolves
        orch.process_evaluation(eval_deep())
        assert orch.state == "GUIDED_PRACTICE"
        assert "M2" in sn["misconceptions_resolved"]


# ===========================================================================
# GP ↔ DIAGNOSE loop breaker
# ===========================================================================

class TestDiagnoseLoopBreaker:
    def test_three_diagnose_failures_triggers_model(self, orch):
        """After 3 consecutive DIAGNOSE visits without deep reasoning → MODEL."""
        sn = orch.student_node
        sn["diagnose_count"] = 0

        # Each cycle: GP → mechanical → DIAGNOSE → mechanical → GP
        # Diagnose visit 1
        orch.state = "DIAGNOSE"
        orch.process_evaluation(eval_mechanical())
        assert sn["diagnose_count"] == 1
        assert orch.state == "GUIDED_PRACTICE"

        # Diagnose visit 2
        orch.state = "DIAGNOSE"
        orch.process_evaluation(eval_mechanical())
        assert sn["diagnose_count"] == 2
        assert orch.state == "GUIDED_PRACTICE"

        # Diagnose visit 3 — loop breaker fires
        orch.state = "DIAGNOSE"
        orch.process_evaluation(eval_mechanical())
        assert sn["diagnose_count"] == 0  # reset after escalation
        assert orch.state == "MODEL"

    def test_two_diagnose_failures_stays_in_gp(self, orch):
        """2 DIAGNOSE failures stay in GP — loop breaker not yet triggered."""
        sn = orch.student_node
        sn["diagnose_count"] = 0

        orch.state = "DIAGNOSE"
        orch.process_evaluation(eval_mechanical())
        assert orch.state == "GUIDED_PRACTICE"

        orch.state = "DIAGNOSE"
        orch.process_evaluation(eval_mechanical())
        assert orch.state == "GUIDED_PRACTICE"
        assert sn["diagnose_count"] == 2

    def test_deep_reasoning_resets_diagnose_counter(self, orch):
        """Deep reasoning in DIAGNOSE resets the counter."""
        sn = orch.student_node
        sn["diagnose_count"] = 2  # one away from triggering

        orch.state = "DIAGNOSE"
        orch.scaffolding_level = 1
        orch.process_evaluation(eval_deep(physical=True))
        assert sn["diagnose_count"] == 0
        assert orch.state == "ASSESS_MASTERY"

    def test_misconception_in_diagnose_resets_counter(self, orch):
        """Misconception detection in DIAGNOSE resets the counter (different path)."""
        sn = orch.student_node
        sn["diagnose_count"] = 2

        orch.state = "DIAGNOSE"
        orch.process_evaluation(eval_major_error(misconception="M1"))
        assert sn["diagnose_count"] == 0
        assert orch.state == "REMEDIATE"

    def test_absent_reasoning_also_increments_counter(self, orch):
        """Absent reasoning in DIAGNOSE also increments the counter."""
        sn = orch.student_node
        sn["diagnose_count"] = 0

        orch.state = "DIAGNOSE"
        orch.process_evaluation(eval_absent_reasoning())
        assert sn["diagnose_count"] == 1
        assert orch.state == "GUIDED_PRACTICE"

    def test_full_loop_break_cycle(self, orch):
        """Simulate the full GP ↔ DIAGNOSE loop and verify it breaks."""
        sn = orch.student_node
        sn["diagnose_count"] = 0
        orch.state = "GUIDED_PRACTICE"

        states = []
        for i in range(10):
            if orch.state == "MODEL":
                states.append("MODEL")
                break
            if orch.state == "GUIDED_PRACTICE":
                orch.process_evaluation(eval_mechanical())
                states.append(f"GP→{orch.state}")
            elif orch.state == "DIAGNOSE":
                orch.process_evaluation(eval_mechanical())
                states.append(f"DIAG→{orch.state}")

        # Should have broken to MODEL before exhausting iterations
        assert "MODEL" in states, f"Loop did not break: {states}"


# ===========================================================================
# Full curriculum progression
# ===========================================================================

MODULE1_NODES = [
    "element_stiffness",
    "local_global_dofs",
    "assembly",
    "boundary_conditions",
    "solution",
    "verification",
]

MODULE2_NODES = [
    "bar_element",
    "two_d_dofs",
    "coordinate_transformation",
    "global_element_stiffness",
    "truss_assembly_solution",
    "truss_force_recovery",
]

MODULE3_NODES = [
    "beam_element",
    "frame_element",
    "frame_transformation",
    "frame_global_stiffness",
    "frame_assembly_solution",
    "frame_force_recovery",
]

ALL_NODES = MODULE1_NODES + MODULE2_NODES + MODULE3_NODES


class TestFullModule1Progression:
    def test_strong_student_advances_through_module1(self, orch):
        """Simulate a strong student advancing through all 6 Module 1 nodes."""
        visited = [orch.current_node]

        for node_name in MODULE1_NODES:
            assert orch.current_node == node_name
            assert orch.state == "ASSESS_PRIOR"

            orch.process_evaluation(eval_deep())
            assert orch.state == "ASSESS_MASTERY"

            orch.process_evaluation(eval_deep())
            assert orch.state == "ADVANCE"

            has_next = orch.advance_to_next_node()
            assert has_next is True  # Module 2 follows
            visited.append(orch.current_node)

        assert visited == MODULE1_NODES + ["bar_element"]


class TestCurriculumCompletion:
    def test_no_advance_after_last_node(self, orch):
        """After mastering frame_force_recovery, advance_to_next_node returns False."""
        orch.student["current_node"] = "frame_force_recovery"
        orch.state = "ADVANCE"
        result = orch.advance_to_next_node()
        assert result is False
        assert orch.current_node == "frame_force_recovery"

    def test_state_unchanged_after_failed_advance(self, orch):
        """advance_to_next_node returning False does not change state."""
        orch.student["current_node"] = "frame_force_recovery"
        orch.state = "ADVANCE"
        orch.advance_to_next_node()
        assert orch.state == "ADVANCE"


class TestProblemMapCompleteness:
    def test_all_nodes_in_problem_map(self, orch):
        """PROBLEM_MAP has entries for all 18 node IDs in the curriculum graph."""
        curriculum_nodes = list(orch.curriculum["nodes"].keys())
        for node_id in curriculum_nodes:
            assert node_id in PROBLEM_MAP, f"Missing PROBLEM_MAP entry for {node_id}"

    def test_all_required_states_per_node(self, orch):
        """Each node in PROBLEM_MAP has keys for all 5 required states."""
        required_states = {"MODEL", "GUIDED_PRACTICE", "DIAGNOSE", "REMEDIATE", "ASSESS_MASTERY"}
        for node_id, mapping in PROBLEM_MAP.items():
            for state in required_states:
                assert state in mapping, f"{node_id} missing {state} in PROBLEM_MAP"

    def test_problem_map_covers_all_nodes(self):
        """PROBLEM_MAP has exactly 18 entries, one per curriculum node."""
        assert len(PROBLEM_MAP) == 18
        assert set(PROBLEM_MAP.keys()) == set(ALL_NODES)


# ===========================================================================
# GP ↔ REMEDIATE loop breaker
# ===========================================================================

class TestRemediateLoopBreaker:
    def test_two_remediation_attempts_triggers_model(self, orch):
        """After 2 remediation attempts for same misconception → MODEL."""
        sn = orch.student_node
        sn["misconceptions_observed"] = ["M2"]
        sn["remediation_attempts"] = {}

        # First remediation attempt — goes back to GP
        orch.state = "REMEDIATE"
        orch.process_evaluation(eval_confused())
        assert orch.state == "GUIDED_PRACTICE"
        assert sn["remediation_attempts"]["M2"] == 1

        # Second remediation attempt — loop breaker fires
        orch.state = "REMEDIATE"
        orch.process_evaluation(eval_confused())
        assert orch.state == "MODEL"
        assert sn["remediation_attempts"]["M2"] == 2

    def test_one_remediation_stays_in_gp(self, orch):
        """1 remediation attempt stays in GP — not yet escalated."""
        sn = orch.student_node
        sn["misconceptions_observed"] = ["M2"]
        sn["remediation_attempts"] = {}

        orch.state = "REMEDIATE"
        orch.process_evaluation(eval_confused())
        assert orch.state == "GUIDED_PRACTICE"
        assert sn["remediation_attempts"]["M2"] == 1

    def test_different_misconceptions_tracked_separately(self, orch):
        """Remediation attempts for different misconceptions are independent."""
        sn = orch.student_node
        sn["misconceptions_observed"] = ["M1"]
        sn["remediation_attempts"] = {}

        # Remediate M1 once
        orch.state = "REMEDIATE"
        orch.process_evaluation(eval_confused())
        assert orch.state == "GUIDED_PRACTICE"
        assert sn["remediation_attempts"]["M1"] == 1

        # Now observe M2
        sn["misconceptions_observed"].append("M2")

        # Remediate M2 once — should go to GP (M2 count is 1)
        orch.state = "REMEDIATE"
        orch.process_evaluation(eval_confused())
        assert orch.state == "GUIDED_PRACTICE"
        assert sn["remediation_attempts"]["M2"] == 1
        assert sn["remediation_attempts"]["M1"] == 1

    def test_remodel_misconception_flag_set(self, orch):
        """When remediation loop breaks, _remodel_misconception flag is set."""
        sn = orch.student_node
        sn["misconceptions_observed"] = ["M2"]
        sn["remediation_attempts"] = {"M2": 1}  # one away

        orch.state = "REMEDIATE"
        orch.process_evaluation(eval_confused())
        assert orch.state == "MODEL"
        assert sn.get("_remodel_misconception") == "M2"

    def test_remodel_misconception_in_teacher_context(self, orch):
        """MODEL context includes remodel_misconception when flag is set."""
        sn = orch.student_node
        sn["_remodel_misconception"] = "M2"

        orch.state = "MODEL"
        ctx = orch.build_teacher_context()
        assert "remodel_misconception" in ctx
        assert ctx["remodel_misconception"]["id"] == "M2"
        assert "singular" in ctx["remodel_misconception"]["description"].lower()
        # Flag should be consumed (popped)
        assert "_remodel_misconception" not in sn

    def test_remodel_misconception_absent_without_flag(self, orch):
        """MODEL context does NOT include remodel_misconception normally."""
        orch.state = "MODEL"
        ctx = orch.build_teacher_context()
        assert "remodel_misconception" not in ctx

    def test_resolution_resets_remediation_counter(self, orch):
        """Resolving a misconception resets its remediation counter."""
        sn = orch.student_node
        sn["misconceptions_observed"] = ["M2"]
        sn["remediation_attempts"] = {"M2": 1}

        # Resolve M2 via deep reasoning on ES_C (targets M2)
        orch.state = "ASSESS_MASTERY"
        orch.process_evaluation(eval_deep())
        assert "M2" in sn["misconceptions_resolved"]
        assert "M2" not in sn.get("remediation_attempts", {})

    def test_full_remediate_loop_break_cycle(self, orch):
        """Simulate the full GP ↔ REMEDIATE loop and verify it breaks."""
        sn = orch.student_node
        sn["misconceptions_observed"] = ["M2"]
        sn["remediation_attempts"] = {}

        states = []
        orch.state = "GUIDED_PRACTICE"
        for i in range(10):
            if orch.state == "MODEL":
                states.append("MODEL")
                break
            if orch.state == "GUIDED_PRACTICE":
                # Trigger misconception → REMEDIATE
                orch.process_evaluation(eval_confused(misconception="M2"))
                states.append(f"GP→{orch.state}")
            elif orch.state == "REMEDIATE":
                orch.process_evaluation(eval_confused())
                states.append(f"REM→{orch.state}")

        assert "MODEL" in states, f"Loop did not break: {states}"


# ===========================================================================
# Full curriculum progression (all modules)
# ===========================================================================

class TestFullEighteenNodeProgression:
    def test_strong_student_advances_through_all_nodes(self, orch):
        """Simulate a strong student advancing through all 18 nodes in sequence,
        including Module 1→2 (verification→bar_element) and Module 2→3
        (truss_force_recovery→beam_element) bridges."""
        visited = [orch.current_node]

        for node_name in ALL_NODES:
            assert orch.current_node == node_name
            assert orch.state == "ASSESS_PRIOR"

            # Turn 1: ASSESS_PRIOR deep → ASSESS_MASTERY
            orch.process_evaluation(eval_deep())
            assert orch.state == "ASSESS_MASTERY"

            # Turn 2: ASSESS_MASTERY deep → ADVANCE (BKT crosses 0.90)
            orch.process_evaluation(eval_deep())
            assert orch.state == "ADVANCE"

            # Advance to next node (or finish)
            has_next = orch.advance_to_next_node()
            if node_name != "frame_force_recovery":
                assert has_next is True
                visited.append(orch.current_node)
            else:
                assert has_next is False

        assert visited == ALL_NODES

    def test_module1_to_module2_bridge(self, orch):
        """The Module 1 → Module 2 transition (verification → bar_element) works."""
        orch.student["current_node"] = "verification"
        orch.state = "ADVANCE"
        result = orch.advance_to_next_node()
        assert result is True
        assert orch.current_node == "bar_element"
        assert orch.state == "ASSESS_PRIOR"
        assert orch.scaffolding_level == 3

    def test_module2_to_module3_bridge(self, orch):
        """The Module 2 → Module 3 transition (truss_force_recovery → beam_element) works."""
        orch.student["current_node"] = "truss_force_recovery"
        orch.state = "ADVANCE"
        result = orch.advance_to_next_node()
        assert result is True
        assert orch.current_node == "beam_element"
        assert orch.state == "ASSESS_PRIOR"
        assert orch.scaffolding_level == 3


class TestModule1ToModule2Transition:
    def test_edge_exists_verification_to_bar_element(self, orch):
        """The curriculum graph has an edge from verification to bar_element."""
        assert ["verification", "bar_element"] in orch.curriculum["edges"]

    def test_advance_from_verification_to_bar_element(self, orch):
        """advance_to_next_node correctly transitions from verification to bar_element."""
        orch.student["current_node"] = "verification"
        orch.state = "ADVANCE"
        result = orch.advance_to_next_node()
        assert result is True
        assert orch.current_node == "bar_element"
        assert orch.state == "ASSESS_PRIOR"
        assert orch.scaffolding_level == 3


# ===========================================================================
# Module 3 — Frame Elements
# ===========================================================================

class TestModule3CurriculumCompletion:
    def test_module3_nodes_reachable(self, orch):
        """Starting from beam_element, all 6 Module 3 nodes can be reached
        by advancing through the sequence."""
        orch.student["current_node"] = "beam_element"
        orch.state = "ASSESS_PRIOR"
        visited = []

        for node_name in MODULE3_NODES:
            assert orch.current_node == node_name
            visited.append(orch.current_node)

            orch.process_evaluation(eval_deep())
            orch.process_evaluation(eval_deep())
            assert orch.state == "ADVANCE"

            has_next = orch.advance_to_next_node()
            if node_name != "frame_force_recovery":
                assert has_next is True
            else:
                assert has_next is False

        assert visited == MODULE3_NODES

    def test_each_module3_node_has_three_problems(self, orch):
        """Each Module 3 node has exactly 3 problems in the problem bank."""
        for node_id in MODULE3_NODES:
            problems = [p for p in orch.problem_bank.values()
                        if p.get("node") == node_id]
            assert len(problems) == 3, f"{node_id} has {len(problems)} problems, expected 3"


class TestModule3ProblemMapCompleteness:
    def test_all_module3_nodes_in_problem_map(self, orch):
        """Every Module 3 node in the graph has a PROBLEM_MAP entry."""
        for node_id in MODULE3_NODES:
            assert node_id in PROBLEM_MAP, f"Missing PROBLEM_MAP entry for {node_id}"

    def test_module3_required_states(self, orch):
        """Each Module 3 node has all 5 required state keys."""
        required_states = {"MODEL", "GUIDED_PRACTICE", "DIAGNOSE", "REMEDIATE", "ASSESS_MASTERY"}
        for node_id in MODULE3_NODES:
            mapping = PROBLEM_MAP[node_id]
            for state in required_states:
                assert state in mapping, f"{node_id} missing {state} in PROBLEM_MAP"

    def test_module3_problems_exist_in_bank(self, orch):
        """Every problem listed in PROBLEM_MAP for Module 3 nodes exists in the bank."""
        for node_id in MODULE3_NODES:
            mapping = PROBLEM_MAP[node_id]
            for state, ref in mapping.items():
                ids = ref if isinstance(ref, list) else [ref]
                for pid in ids:
                    assert pid in orch.problem_bank, \
                        f"{node_id}/{state}: problem {pid} not found in bank"


class TestModule2ToModule3Transition:
    def test_edge_exists_truss_to_beam(self, orch):
        """The curriculum graph has an edge from truss_force_recovery to beam_element."""
        assert ["truss_force_recovery", "beam_element"] in orch.curriculum["edges"]

    def test_advance_from_truss_to_beam(self, orch):
        """advance_to_next_node correctly transitions from truss_force_recovery to beam_element."""
        orch.student["current_node"] = "truss_force_recovery"
        orch.state = "ADVANCE"
        result = orch.advance_to_next_node()
        assert result is True
        assert orch.current_node == "beam_element"
        assert orch.state == "ASSESS_PRIOR"
        assert orch.scaffolding_level == 3
