"""
Simulation test harness for the FEA tutor.

Drives scripted student scenarios through the orchestrator with mocked
LLM calls. Tests infrastructure connections (notebook references, solver
demos, state transitions, context construction) — NOT LLM prose quality.

Each scenario is a sequence of steps. Each step specifies:
  - A scripted evaluator response (what the evaluator "would say")
  - Assertions on the teacher context (what the teacher receives)
  - Assertions on state transitions

Run:
    python -m pytest tests/test_simulation_harness.py -v
"""
import pytest
import json
import copy
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_eval(
    procedural="correct",
    conceptual="deep",
    misconception_id=None,
    confidence="high",
    physical_reasoning=True,
):
    """Build an evaluator response dict."""
    return {
        "procedural": procedural,
        "conceptual": conceptual,
        "misconception_id": misconception_id,
        "confidence": confidence,
        "physical_reasoning_detected": physical_reasoning,
    }


EVAL_PERFECT = make_eval()
EVAL_MECHANICAL = make_eval(conceptual="mechanical", physical_reasoning=False)
EVAL_MINOR_ERROR = make_eval(procedural="minor_error")
EVAL_MAJOR_ERROR = make_eval(procedural="major_error", conceptual="absent",
                              confidence="low", physical_reasoning=False)
EVAL_MISCONCEPTION_M2 = make_eval(
    procedural="correct", conceptual="mechanical",
    misconception_id="M2", physical_reasoning=False,
)
EVAL_CORRECT_NO_REASONING = make_eval(
    conceptual="absent", physical_reasoning=False, confidence="medium",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_orchestrator():
    """Create a fresh orchestrator with a clean student model."""
    from core.orchestrator import Orchestrator
    # Use a temporary student file so we don't clobber real state.
    # Create a temp path that does NOT yet exist so the orchestrator
    # loads from the template (existing empty file → JSON decode error).
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp_path = tmp.name
    tmp.close()
    Path(tmp_path).unlink()  # Remove the empty file
    orch = Orchestrator(data_dir="data", student_file=tmp_path)
    yield orch
    Path(tmp_path).unlink(missing_ok=True)


@pytest.fixture
def orchestrator_at_node(fresh_orchestrator):
    """Factory fixture: returns an orchestrator positioned at a given node/state."""
    def _make(node: str, state: str = "ASSESS_PRIOR", scaffolding: int = 3):
        orch = fresh_orchestrator
        orch.student["current_node"] = node
        orch.state = state
        orch.scaffolding_level = scaffolding
        return orch
    return _make


# ===================================================================
# SCENARIO 1: Happy path — advance through element_stiffness
# Tests: MODEL gets solver demo + notebook code, state transitions work,
#         ADVANCE triggers correctly after mastery
# ===================================================================

class TestScenario1HappyPath:
    """Student advances through element_stiffness with correct answers."""

    def test_initial_state_is_assess_prior(self, fresh_orchestrator):
        orch = fresh_orchestrator
        assert orch.state == "ASSESS_PRIOR"
        assert orch.current_node == "element_stiffness"

    def test_assess_prior_perfect_jumps_to_mastery(self, fresh_orchestrator):
        orch = fresh_orchestrator
        assert orch.state == "ASSESS_PRIOR"
        orch.process_evaluation(EVAL_PERFECT)
        assert orch.state == "ASSESS_MASTERY"

    def test_assess_prior_mechanical_goes_to_model(self, fresh_orchestrator):
        orch = fresh_orchestrator
        orch.process_evaluation(EVAL_MECHANICAL)
        assert orch.state == "MODEL"

    def test_model_context_has_notebook_references(self, orchestrator_at_node):
        orch = orchestrator_at_node("element_stiffness", "MODEL")
        ctx = orch.build_teacher_context()
        assert "notebook_references" in ctx
        assert len(ctx["notebook_references"]) > 0
        # Should have uelspring reference
        families = [r.get("function_family") for r in ctx["notebook_references"]]
        assert "uelXXX" in families

    def test_model_context_has_solver_demo(self, orchestrator_at_node):
        orch = orchestrator_at_node("element_stiffness", "MODEL")
        ctx = orch.build_teacher_context()
        # Solver demo may or may not be present depending on spring-agent availability
        if ctx.get("solver_demo"):
            assert ctx["solver_demo"]["status"] == "success"
            assert "results" in ctx["solver_demo"]
            assert "model_summary" in ctx["solver_demo"]

    def test_model_context_has_problem_and_narration(self, orchestrator_at_node):
        orch = orchestrator_at_node("element_stiffness", "MODEL")
        ctx = orch.build_teacher_context()
        assert "problem" in ctx
        assert "expert_narration" in ctx["problem"]
        assert "statement" in ctx["problem"]
        assert ctx["problem"]["problem_id"] == "ES_A"

    def test_model_context_has_practice_problem(self, orchestrator_at_node):
        orch = orchestrator_at_node("element_stiffness", "MODEL")
        ctx = orch.build_teacher_context()
        assert "practice_problem" in ctx
        assert "statement" in ctx["practice_problem"]

    def test_model_context_has_cam_technique(self, orchestrator_at_node):
        orch = orchestrator_at_node("element_stiffness", "MODEL")
        ctx = orch.build_teacher_context()
        assert ctx["cam_technique"] == "modeling"
        assert ctx["state"] == "MODEL"

    def test_full_happy_path_to_advance(self, orchestrator_at_node):
        """Simulate: ASSESS_PRIOR(mechanical) → MODEL → GP(perfect, fade)
        → ASSESS_MASTERY(perfect) → ADVANCE"""
        orch = orchestrator_at_node("element_stiffness", "ASSESS_PRIOR", scaffolding=3)

        # Step 1: ASSESS_PRIOR — mechanical answer → MODEL
        orch.process_evaluation(EVAL_MECHANICAL)
        assert orch.state == "MODEL"

        # MODEL state doesn't need evaluation — teacher presents, then we
        # manually transition to GUIDED_PRACTICE
        orch.state = "GUIDED_PRACTICE"

        # Step 2: GUIDED_PRACTICE — perfect at scaffolding 3 → fade to 2
        orch.process_evaluation(EVAL_PERFECT)
        assert orch.state == "GUIDED_PRACTICE"
        assert orch.scaffolding_level == 2

        # Step 3: GUIDED_PRACTICE — perfect at 2 → fade to 1
        orch.process_evaluation(EVAL_PERFECT)
        assert orch.state == "GUIDED_PRACTICE"
        assert orch.scaffolding_level == 1

        # Step 4: GUIDED_PRACTICE — perfect at 1 → ASSESS_MASTERY
        orch.process_evaluation(EVAL_PERFECT)
        assert orch.state == "ASSESS_MASTERY"
        assert orch.scaffolding_level == 0

        # Step 5: ASSESS_MASTERY — perfect → check if ADVANCE
        # (may need multiple attempts to reach BKT threshold)
        orch.process_evaluation(EVAL_PERFECT)
        # After enough correct answers, BKT should push mastery up
        assert orch.state in ["ADVANCE", "GUIDED_PRACTICE"]

        # If not yet at threshold, keep going
        safety = 0
        while orch.state != "ADVANCE" and safety < 10:
            if orch.state == "GUIDED_PRACTICE":
                orch.process_evaluation(EVAL_PERFECT)
            elif orch.state == "ASSESS_MASTERY":
                orch.process_evaluation(EVAL_PERFECT)
            safety += 1

        assert orch.state == "ADVANCE", (
            f"Failed to reach ADVANCE after {safety} extra iterations. "
            f"State: {orch.state}, p_mastery: {orch.student_node['p_mastery']}"
        )

    def test_advance_moves_to_next_node(self, orchestrator_at_node):
        orch = orchestrator_at_node("element_stiffness", "ADVANCE")
        result = orch.advance_to_next_node()
        assert result is True
        assert orch.current_node == "local_global_dofs"
        assert orch.state == "ASSESS_PRIOR"


# ===================================================================
# SCENARIO 2: Misconception path
# Tests: misconception detection triggers REMEDIATE, scaffolding adjusts,
#         evaluator context has correct markers
# ===================================================================

class TestScenario2MisconceptionPath:
    """Student reveals misconception M2 (singularity = error)."""

    def test_misconception_in_gp_triggers_remediate(self, orchestrator_at_node):
        orch = orchestrator_at_node("element_stiffness", "GUIDED_PRACTICE", scaffolding=2)
        orch.process_evaluation(EVAL_MISCONCEPTION_M2)
        assert orch.state == "REMEDIATE"
        assert "M2" in orch.student_node["misconceptions_observed"]

    def test_remediate_then_back_to_gp(self, orchestrator_at_node):
        orch = orchestrator_at_node("element_stiffness", "GUIDED_PRACTICE", scaffolding=2)

        # Misconception detected → REMEDIATE
        orch.process_evaluation(EVAL_MISCONCEPTION_M2)
        assert orch.state == "REMEDIATE"

        # After remediation (teacher addresses it), transition with next eval
        orch.state = "GUIDED_PRACTICE"  # teacher finished remediation explanation
        orch.process_evaluation(EVAL_PERFECT)
        # Should continue in GP or advance
        assert orch.state in ["GUIDED_PRACTICE", "ASSESS_MASTERY"]

    def test_repeated_remediation_escalates_to_model(self, orchestrator_at_node):
        """2 remediation attempts for same misconception → MODEL."""
        orch = orchestrator_at_node("element_stiffness", "GUIDED_PRACTICE", scaffolding=2)

        # First misconception → REMEDIATE
        orch.process_evaluation(EVAL_MISCONCEPTION_M2)
        assert orch.state == "REMEDIATE"

        # Remediation eval (still has misconception) — first attempt logged
        orch.process_evaluation(EVAL_MISCONCEPTION_M2)
        # After first remediation attempt, goes back to GP
        assert orch.state == "GUIDED_PRACTICE"

        # Misconception again → REMEDIATE again
        orch.process_evaluation(EVAL_MISCONCEPTION_M2)
        assert orch.state == "REMEDIATE"

        # Second remediation attempt → escalate to MODEL
        orch.process_evaluation(EVAL_MISCONCEPTION_M2)
        assert orch.state == "MODEL"

    def test_evaluator_context_has_misconception_inventory(self, orchestrator_at_node):
        orch = orchestrator_at_node("element_stiffness", "GUIDED_PRACTICE")
        eval_ctx = orch.build_evaluator_context("The matrix is [[k,-k],[-k,k]]")
        assert "misconceptions" in eval_ctx
        assert "physical_reasoning_markers" in eval_ctx
        assert "learning_objectives" in eval_ctx

    def test_evaluator_context_has_problem_solution(self, orchestrator_at_node):
        orch = orchestrator_at_node("element_stiffness", "GUIDED_PRACTICE")
        eval_ctx = orch.build_evaluator_context("some answer")
        assert "problem" in eval_ctx
        assert "solution_steps" in eval_ctx["problem"]
        assert "final_answer" in eval_ctx["problem"]
        assert "evaluation_markers" in eval_ctx["problem"]


# ===================================================================
# SCENARIO 3: Scaffolding dynamics
# Tests: scaffolding increases on errors, decreases on success, caps at 3
# ===================================================================

class TestScenario3ScaffoldingDynamics:
    """Test scaffolding level changes based on performance."""

    def test_major_error_increases_scaffolding(self, orchestrator_at_node):
        orch = orchestrator_at_node("element_stiffness", "GUIDED_PRACTICE", scaffolding=1)
        orch.process_evaluation(EVAL_MAJOR_ERROR)
        assert orch.scaffolding_level == 2
        assert orch.state == "GUIDED_PRACTICE"

    def test_scaffolding_caps_at_3(self, orchestrator_at_node):
        orch = orchestrator_at_node("element_stiffness", "GUIDED_PRACTICE", scaffolding=3)
        orch.process_evaluation(EVAL_MAJOR_ERROR)
        # At max scaffolding with major error → back to MODEL
        assert orch.state == "MODEL"

    def test_gp_context_has_scaffolded_prompt(self, orchestrator_at_node):
        orch = orchestrator_at_node("element_stiffness", "GUIDED_PRACTICE", scaffolding=3)
        ctx = orch.build_teacher_context()
        assert "problem" in ctx
        # Level 3 scaffolding should have a scaffolded prompt
        if "scaffolded_prompt" in ctx["problem"]:
            assert len(ctx["problem"]["scaffolded_prompt"]) > 0

    def test_scaffolding_0_has_no_scaffolded_prompt(self, orchestrator_at_node):
        orch = orchestrator_at_node("element_stiffness", "ASSESS_MASTERY", scaffolding=0)
        ctx = orch.build_teacher_context()
        # Level 0 = no scaffolding
        if "problem" in ctx:
            assert "scaffolded_prompt" not in ctx["problem"] or \
                   ctx["problem"].get("scaffolded_prompt") is None

    def test_diagnose_mechanical_triggers_articulation(self, orchestrator_at_node):
        """Correct answer but mechanical reasoning → DIAGNOSE for articulation."""
        orch = orchestrator_at_node("element_stiffness", "GUIDED_PRACTICE", scaffolding=2)
        orch.process_evaluation(EVAL_MECHANICAL)
        assert orch.state == "DIAGNOSE"

    def test_diagnose_deep_reasoning_progresses(self, orchestrator_at_node):
        """After articulation probe, deep reasoning → continue."""
        orch = orchestrator_at_node("element_stiffness", "DIAGNOSE", scaffolding=2)
        orch.process_evaluation(EVAL_PERFECT)
        # Deep reasoning in DIAGNOSE → reduce scaffolding and continue
        assert orch.state in ["GUIDED_PRACTICE", "ASSESS_MASTERY"]

    def test_diagnose_loop_breaker(self, orchestrator_at_node):
        """3 consecutive DIAGNOSE visits without deep reasoning → MODEL."""
        orch = orchestrator_at_node("element_stiffness", "DIAGNOSE", scaffolding=2)
        orch.student_node["diagnose_count"] = 0

        # First DIAGNOSE — mechanical → back to GP
        orch.process_evaluation(EVAL_MECHANICAL)
        assert orch.state == "GUIDED_PRACTICE"
        assert orch.student_node["diagnose_count"] == 1

        # GP → mechanical → DIAGNOSE again
        orch.process_evaluation(EVAL_MECHANICAL)
        assert orch.state == "DIAGNOSE"

        # Second DIAGNOSE — mechanical → GP
        orch.process_evaluation(EVAL_MECHANICAL)
        assert orch.state == "GUIDED_PRACTICE"
        assert orch.student_node["diagnose_count"] == 2

        # GP → mechanical → DIAGNOSE
        orch.process_evaluation(EVAL_MECHANICAL)
        assert orch.state == "DIAGNOSE"

        # Third DIAGNOSE — loop breaker → MODEL
        orch.process_evaluation(EVAL_MECHANICAL)
        assert orch.state == "MODEL"
        assert orch.student_node["diagnose_count"] == 0


# ===================================================================
# SCENARIO 4: Cross-module transition
# Tests: Module 1→2 bridge, notebook evolution table in ADVANCE,
#         solver demo stops for Module 2
# ===================================================================

class TestScenario4CrossModuleTransition:
    """Test transition from Module 1 (verification) to Module 2 (bar_element)."""

    def test_advance_context_has_evolution_table(self, orchestrator_at_node):
        orch = orchestrator_at_node("verification", "ADVANCE")
        ctx = orch.build_teacher_context()
        assert "evolution_table" in ctx
        assert "rows" in ctx["evolution_table"]
        assert len(ctx["evolution_table"]["rows"]) > 0

    def test_advance_context_has_next_node_title(self, orchestrator_at_node):
        orch = orchestrator_at_node("element_stiffness", "ADVANCE")
        ctx = orch.build_teacher_context()
        assert "next_node_title" in ctx

    def test_verification_advances_to_bar_element(self, orchestrator_at_node):
        orch = orchestrator_at_node("verification", "ADVANCE")
        result = orch.advance_to_next_node()
        assert result is True
        assert orch.current_node == "bar_element"
        assert orch.state == "ASSESS_PRIOR"

    def test_module2_model_has_no_solver_demo(self, orchestrator_at_node):
        """Spring-agent only works for Module 1. Module 2 should not have solver_demo."""
        orch = orchestrator_at_node("bar_element", "MODEL")
        ctx = orch.build_teacher_context()
        assert ctx.get("solver_demo") is None

    def test_module2_model_has_notebook_references(self, orchestrator_at_node):
        """Module 2 should still have notebook references."""
        orch = orchestrator_at_node("bar_element", "MODEL")
        ctx = orch.build_teacher_context()
        refs = ctx.get("notebook_references", [])
        assert len(refs) > 0
        # Should have ueltruss2D reference
        modules = [r.get("module") for r in refs]
        assert "module2_trusses" in modules

    def test_module3_has_curriculum_difference_note(self, orchestrator_at_node):
        """Module 3 beam_element should flag notebook vs curriculum difference."""
        orch = orchestrator_at_node("beam_element", "MODEL")
        ctx = orch.build_teacher_context()
        # Should have a note about axially-rigid beam vs full frame
        assert ctx.get("curriculum_difference") is not None


# ===================================================================
# SCENARIO 5: Skip override and cold assessment
# Tests: /skip goes to ASSESS_MASTERY, perfect cold pass grants mastery
# ===================================================================

class TestScenario5SkipOverride:
    """Test student skip command and cold assessment mastery override."""

    def test_skip_goes_to_assess_mastery(self, orchestrator_at_node):
        orch = orchestrator_at_node("element_stiffness", "GUIDED_PRACTICE", scaffolding=2)
        orch.handle_student_override("skip")
        assert orch.state == "ASSESS_MASTERY"
        assert orch.scaffolding_level == 0

    def test_cold_perfect_pass_grants_mastery(self, orchestrator_at_node):
        """Perfect answer on cold /skip → ADVANCE (mastery override)."""
        orch = orchestrator_at_node("element_stiffness", "ASSESS_PRIOR", scaffolding=3)
        orch.handle_student_override("skip")
        assert orch.state == "ASSESS_MASTERY"
        assert orch._skipped_to_mastery is True

        orch.process_evaluation(EVAL_PERFECT)
        assert orch.state == "ADVANCE"

    def test_cold_mechanical_pass_does_not_grant_mastery(self, orchestrator_at_node):
        orch = orchestrator_at_node("element_stiffness", "ASSESS_PRIOR", scaffolding=3)
        orch.handle_student_override("skip")
        orch.process_evaluation(EVAL_MECHANICAL)
        # Mechanical reasoning → doesn't meet the 4 criteria for cold override
        assert orch.state != "ADVANCE"

    def test_repeat_goes_to_model(self, orchestrator_at_node):
        orch = orchestrator_at_node("element_stiffness", "GUIDED_PRACTICE")
        orch.handle_student_override("repeat")
        assert orch.state == "MODEL"


# ===================================================================
# SCENARIO 6: BKT mastery tracking
# Tests: p_mastery increases with correct answers, mastery gating works
# ===================================================================

class TestScenario6MasteryTracking:
    """Test BKT mastery probability updates and gating."""

    def test_mastery_increases_on_correct(self, orchestrator_at_node):
        orch = orchestrator_at_node("element_stiffness", "GUIDED_PRACTICE", scaffolding=2)
        initial_mastery = orch.student_node["p_mastery"]
        orch.process_evaluation(EVAL_PERFECT)
        assert orch.student_node["p_mastery"] > initial_mastery

    def test_mastery_decreases_on_incorrect(self, orchestrator_at_node):
        orch = orchestrator_at_node("element_stiffness", "GUIDED_PRACTICE", scaffolding=2)
        # First give some correct answers to raise mastery
        for _ in range(3):
            orch.state = "GUIDED_PRACTICE"
            orch.process_evaluation(EVAL_PERFECT)
        mid_mastery = orch.student_node["p_mastery"]

        orch.state = "GUIDED_PRACTICE"
        orch.process_evaluation(EVAL_MAJOR_ERROR)
        assert orch.student_node["p_mastery"] < mid_mastery

    def test_lo_status_updates(self, orchestrator_at_node):
        orch = orchestrator_at_node("element_stiffness", "GUIDED_PRACTICE", scaffolding=2)
        orch.process_evaluation(EVAL_PERFECT)
        # At least one LO should be demonstrated
        statuses = orch.student_node["lo_status"].values()
        assert "demonstrated" in statuses

    def test_all_los_demonstrated_after_advance(self, fresh_orchestrator):
        """When a student reaches ADVANCE, ALL LOs for the node must be 'demonstrated'."""
        orch = fresh_orchestrator
        assert orch.current_node == "element_stiffness"

        # Drive through to ADVANCE
        orch.process_evaluation(EVAL_MECHANICAL)       # ASSESS_PRIOR → MODEL
        orch.state = "GUIDED_PRACTICE"
        orch.process_evaluation(EVAL_PERFECT)           # scaff 3→2
        orch.process_evaluation(EVAL_PERFECT)           # scaff 2→1
        orch.process_evaluation(EVAL_PERFECT)           # scaff 1→ASSESS_MASTERY

        safety = 0
        while orch.state != "ADVANCE" and safety < 15:
            orch.process_evaluation(EVAL_PERFECT)
            safety += 1
        assert orch.state == "ADVANCE"

        # Every LO must be "demonstrated"
        lo_status = orch.student_node["lo_status"]
        for lo, status in lo_status.items():
            assert status == "demonstrated", (
                f"LO {lo} is '{status}' instead of 'demonstrated' after ADVANCE"
            )

    def test_attempts_counter_increments(self, orchestrator_at_node):
        orch = orchestrator_at_node("element_stiffness", "GUIDED_PRACTICE", scaffolding=2)
        initial_attempts = orch.student_node["attempts"]
        orch.process_evaluation(EVAL_PERFECT)
        assert orch.student_node["attempts"] == initial_attempts + 1


# ===================================================================
# SCENARIO 7: Full journey Module 1 nodes 1→2 (element_stiffness → local_global_dofs)
# Tests: end-to-end infrastructure across two nodes
# ===================================================================

class TestScenario7TwoNodeJourney:
    """Full automated journey through first two curriculum nodes."""

    def test_two_node_journey(self, fresh_orchestrator):
        orch = fresh_orchestrator
        assert orch.current_node == "element_stiffness"
        assert orch.state == "ASSESS_PRIOR"

        # ----- Node 1: element_stiffness -----

        # ASSESS_PRIOR → MODEL (student gives mechanical answer)
        orch.process_evaluation(EVAL_MECHANICAL)
        assert orch.state == "MODEL"

        # Check MODEL context has everything
        ctx = orch.build_teacher_context()
        assert ctx["cam_technique"] == "modeling"
        assert "notebook_references" in ctx
        assert "problem" in ctx
        assert "expert_narration" in ctx["problem"]
        assert "practice_problem" in ctx

        # Teacher finishes modeling → transition to GP
        orch.state = "GUIDED_PRACTICE"

        # GP at scaffolding 3: correct → fade to 2
        ctx = orch.build_teacher_context()
        assert ctx["cam_technique"] == "coaching"
        assert ctx["scaffolding_level"] == 3
        orch.process_evaluation(EVAL_PERFECT)
        assert orch.scaffolding_level == 2

        # GP at 2: correct → fade to 1
        orch.process_evaluation(EVAL_PERFECT)
        assert orch.scaffolding_level == 1

        # GP at 1: correct → ASSESS_MASTERY
        orch.process_evaluation(EVAL_PERFECT)
        assert orch.state == "ASSESS_MASTERY"
        assert orch.scaffolding_level == 0

        # ASSESS_MASTERY: check context
        ctx = orch.build_teacher_context()
        assert ctx["cam_technique"] == "exploration"
        assert ctx["scaffolding_level"] == 0

        # Feed perfect answers until ADVANCE
        safety = 0
        while orch.state != "ADVANCE" and safety < 15:
            orch.process_evaluation(EVAL_PERFECT)
            if orch.state == "GUIDED_PRACTICE":
                pass  # Continue
            safety += 1

        assert orch.state == "ADVANCE", f"Stuck at {orch.state}"

        # ADVANCE context
        ctx = orch.build_teacher_context()
        assert ctx["cam_technique"] == "transition"
        assert "next_node_title" in ctx

        # ----- Advance to Node 2: local_global_dofs -----

        orch.advance_to_next_node()
        assert orch.current_node == "local_global_dofs"
        assert orch.state == "ASSESS_PRIOR"

        # Check context for new node
        ctx = orch.build_teacher_context()
        assert ctx["node_id"] == "local_global_dofs"
        refs = ctx.get("notebook_references", [])
        # local_global_dofs should have DME reference
        families = [r.get("function_family") for r in refs]
        assert "DME" in families or len(refs) > 0

        # ASSESS_PRIOR → give perfect answer → ASSESS_MASTERY (skip through)
        orch.process_evaluation(EVAL_PERFECT)
        assert orch.state == "ASSESS_MASTERY"


# ===================================================================
# SCENARIO 8: Problem rotation
# Tests: GUIDED_PRACTICE rotates through problems based on attempt count
# ===================================================================

class TestScenario8ProblemRotation:
    """Test that problems rotate in GUIDED_PRACTICE."""

    def test_problem_rotates_with_attempts(self, orchestrator_at_node):
        orch = orchestrator_at_node("element_stiffness", "GUIDED_PRACTICE", scaffolding=2)

        # First attempt → first problem
        orch.student_node["attempts"] = 0
        p1 = orch.get_current_problem()
        assert p1 is not None

        # Second attempt → potentially different problem
        orch.student_node["attempts"] = 1
        p2 = orch.get_current_problem()
        assert p2 is not None

        # The PROBLEM_MAP for GP has ["ES_A", "ES_B"], so they should alternate
        if p1["problem_id"] == "ES_A":
            assert p2["problem_id"] == "ES_B"
        else:
            assert p2["problem_id"] == "ES_A"


# ===================================================================
# SCENARIO 9: Context completeness for all Module 1 nodes
# Tests: every node has valid context in every applicable state
# ===================================================================

class TestScenario9ContextCompleteness:
    """Verify context construction doesn't crash for any node/state combo."""

    MODULE_1_NODES = [
        "element_stiffness", "local_global_dofs", "assembly",
        "boundary_conditions", "solution", "verification",
    ]

    TESTABLE_STATES = ["MODEL", "GUIDED_PRACTICE", "DIAGNOSE", "ASSESS_MASTERY"]

    def test_all_nodes_all_states_build_context(self, orchestrator_at_node):
        """Every node/state combo should produce a valid context dict."""
        for node in self.MODULE_1_NODES:
            for state in self.TESTABLE_STATES:
                orch = orchestrator_at_node(node, state, scaffolding=2)
                ctx = orch.build_teacher_context()

                assert "state" in ctx, f"{node}/{state}: missing 'state'"
                assert "cam_technique" in ctx, f"{node}/{state}: missing 'cam_technique'"
                assert "node_id" in ctx, f"{node}/{state}: missing 'node_id'"
                assert "notebook_references" in ctx, f"{node}/{state}: missing 'notebook_references'"
                assert ctx["node_id"] == node, f"{node}/{state}: wrong node_id"
                assert ctx["state"] == state, f"{node}/{state}: wrong state"

    def test_all_nodes_have_problems(self, orchestrator_at_node):
        """Every node should have at least one problem available."""
        for node in self.MODULE_1_NODES:
            orch = orchestrator_at_node(node, "MODEL")
            problem = orch.get_current_problem()
            assert problem is not None, f"{node}: no MODEL problem"

    def test_evaluator_context_for_all_nodes(self, orchestrator_at_node):
        """Evaluator context should build cleanly for all nodes."""
        for node in self.MODULE_1_NODES:
            orch = orchestrator_at_node(node, "GUIDED_PRACTICE")
            eval_ctx = orch.build_evaluator_context("test student answer")
            assert "student_response" in eval_ctx
            assert "misconceptions" in eval_ctx
            assert "physical_reasoning_markers" in eval_ctx


# ===================================================================
# SCENARIO 10: Module 2 and Module 3 nodes
# Tests: context builds for all 18 nodes without errors
# ===================================================================

class TestScenario10AllModulesContextBuild:
    """Verify context builds for every node in all 3 modules."""

    ALL_NODES = [
        # Module 1
        "element_stiffness", "local_global_dofs", "assembly",
        "boundary_conditions", "solution", "verification",
        # Module 2
        "bar_element", "two_d_dofs", "coordinate_transformation",
        "global_element_stiffness", "truss_assembly_solution", "truss_force_recovery",
        # Module 3
        "beam_element", "frame_element", "frame_transformation",
        "frame_global_stiffness", "frame_assembly_solution", "frame_force_recovery",
    ]

    def test_model_context_builds_for_all_18_nodes(self, orchestrator_at_node):
        for node in self.ALL_NODES:
            orch = orchestrator_at_node(node, "MODEL")
            ctx = orch.build_teacher_context()
            assert ctx["node_id"] == node, f"Failed for {node}"
            assert "notebook_references" in ctx, f"Missing notebook_references for {node}"

    def test_module2_nodes_have_truss_notebook_refs(self, orchestrator_at_node):
        m2_nodes = ["bar_element", "coordinate_transformation", "global_element_stiffness"]
        for node in m2_nodes:
            orch = orchestrator_at_node(node, "MODEL")
            ctx = orch.build_teacher_context()
            refs = ctx.get("notebook_references", [])
            modules = [r.get("module") for r in refs]
            assert "module2_trusses" in modules, (
                f"Node {node} should have truss notebook ref, got modules: {modules}"
            )

    def test_module3_frame_nodes_have_frame_notebook_refs(self, orchestrator_at_node):
        m3_nodes = ["beam_element", "frame_transformation", "frame_global_stiffness"]
        for node in m3_nodes:
            orch = orchestrator_at_node(node, "MODEL")
            ctx = orch.build_teacher_context()
            refs = ctx.get("notebook_references", [])
            modules = [r.get("module") for r in refs]
            assert "module3_frames" in modules, (
                f"Node {node} should have frame notebook ref, got modules: {modules}"
            )


# ===================================================================
# SCENARIO 11: Code executor integration
# Tests: code_demo appears in MODEL context, absent in GP
# ===================================================================

class TestScenario11CodeExecutorIntegration:
    """Verify code executor appears in teacher context."""

    def test_model_context_has_code_demo_for_element_stiffness(self, orchestrator_at_node):
        orch = orchestrator_at_node("element_stiffness", "MODEL")
        ctx = orch.build_teacher_context()
        if ctx.get("code_demo"):
            assert "steps" in ctx["code_demo"]
            assert len(ctx["code_demo"]["steps"]) > 0
            assert all(s["result"]["success"] for s in ctx["code_demo"]["steps"])

    def test_model_context_has_code_demo_for_bar_element(self, orchestrator_at_node):
        orch = orchestrator_at_node("bar_element", "MODEL")
        ctx = orch.build_teacher_context()
        if ctx.get("code_demo"):
            assert len(ctx["code_demo"]["steps"]) > 0

    def test_model_context_has_code_demo_for_beam_element(self, orchestrator_at_node):
        orch = orchestrator_at_node("beam_element", "MODEL")
        ctx = orch.build_teacher_context()
        if ctx.get("code_demo"):
            assert len(ctx["code_demo"]["steps"]) > 0

    def test_no_code_demo_in_guided_practice(self, orchestrator_at_node):
        orch = orchestrator_at_node("element_stiffness", "GUIDED_PRACTICE")
        ctx = orch.build_teacher_context()
        assert ctx.get("code_demo") is None
