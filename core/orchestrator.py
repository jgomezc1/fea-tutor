"""
Pedagogical Orchestrator — the deterministic state machine that drives the
CAM (Cognitive Apprenticeship Model) teaching cycle.

States:
    ASSESS_PRIOR  → Determine what the student already knows
    MODEL         → Expert demonstration with visible reasoning
    GUIDED_PRACTICE → Student attempts with coaching/scaffolding
    DIAGNOSE      → Evaluate student performance (may involve articulation)
    REMEDIATE     → Address specific misconceptions
    ASSESS_MASTERY → Level-0 assessment (no scaffolding)
    ADVANCE       → Transition to next concept node

The orchestrator does NOT use an LLM. It is pure Python logic that reads the
student model and evaluator output to decide state transitions and construct
context for the teacher/evaluator agents.
"""

import json
import copy
from pathlib import Path
from core.bkt import update_mastery, check_mastery
from core.notebook_knowledge import NotebookKnowledge
from core.notebook_executor import NotebookExecutor, NODE_TO_DEMOS


# Valid states
STATES = [
    "ASSESS_PRIOR",
    "MODEL",
    "GUIDED_PRACTICE",
    "DIAGNOSE",
    "REMEDIATE",
    "ASSESS_MASTERY",
    "ADVANCE",
]

# Problem selection mapping: which problem to use in which state
# For each node, maps state to problem_id
PROBLEM_MAP = {
    "element_stiffness": {
        "MODEL": "ES_A",
        "GUIDED_PRACTICE": ["ES_A", "ES_B"],
        "DIAGNOSE": "ES_B",
        "REMEDIATE": "ES_C",
        "ASSESS_MASTERY": "ES_C",
    },
    "local_global_dofs": {
        "MODEL": "DOF_A",
        "GUIDED_PRACTICE": ["DOF_A", "DOF_B"],
        "DIAGNOSE": "DOF_B",
        "REMEDIATE": "DOF_B",
        "ASSESS_MASTERY": "DOF_C",
    },
    "assembly": {
        "MODEL": "ASM_A",
        "GUIDED_PRACTICE": ["ASM_A", "ASM_B"],
        "DIAGNOSE": "ASM_B",
        "REMEDIATE": "ASM_A",
        "ASSESS_MASTERY": "ASM_C",
    },
    "boundary_conditions": {
        "MODEL": "BC_A",
        "GUIDED_PRACTICE": ["BC_A", "BC_B"],
        "DIAGNOSE": "BC_B",
        "REMEDIATE": "BC_A",
        "ASSESS_MASTERY": "BC_C",
    },
    "solution": {
        "MODEL": "SOL_A",
        "GUIDED_PRACTICE": ["SOL_A", "SOL_B"],
        "DIAGNOSE": "SOL_B",
        "REMEDIATE": "SOL_A",
        "ASSESS_MASTERY": "SOL_C",
    },
    "verification": {
        "MODEL": "VER_A",
        "GUIDED_PRACTICE": ["VER_A", "VER_B"],
        "DIAGNOSE": "VER_B",
        "REMEDIATE": "VER_A",
        "ASSESS_MASTERY": "VER_C",
    },
    "bar_element": {
        "MODEL": "BAR_A",
        "GUIDED_PRACTICE": ["BAR_A", "BAR_B"],
        "DIAGNOSE": "BAR_B",
        "REMEDIATE": "BAR_A",
        "ASSESS_MASTERY": "BAR_C",
    },
    "two_d_dofs": {
        "MODEL": "TDOF_A",
        "GUIDED_PRACTICE": ["TDOF_A", "TDOF_B"],
        "DIAGNOSE": "TDOF_B",
        "REMEDIATE": "TDOF_A",
        "ASSESS_MASTERY": "TDOF_C",
    },
    "coordinate_transformation": {
        "MODEL": "CT_A",
        "GUIDED_PRACTICE": ["CT_A", "CT_B"],
        "DIAGNOSE": "CT_B",
        "REMEDIATE": "CT_A",
        "ASSESS_MASTERY": "CT_C",
    },
    "global_element_stiffness": {
        "MODEL": "GES_A",
        "GUIDED_PRACTICE": ["GES_A", "GES_B"],
        "DIAGNOSE": "GES_B",
        "REMEDIATE": "GES_A",
        "ASSESS_MASTERY": "GES_C",
    },
    "truss_assembly_solution": {
        "MODEL": "TAS_A",
        "GUIDED_PRACTICE": ["TAS_A", "TAS_B"],
        "DIAGNOSE": "TAS_B",
        "REMEDIATE": "TAS_A",
        "ASSESS_MASTERY": "TAS_C",
    },
    "truss_force_recovery": {
        "MODEL": "TFR_A",
        "GUIDED_PRACTICE": ["TFR_A", "TFR_B"],
        "DIAGNOSE": "TFR_B",
        "REMEDIATE": "TFR_A",
        "ASSESS_MASTERY": "TFR_C",
    },
    "beam_element": {
        "MODEL": "BEAM_A",
        "GUIDED_PRACTICE": ["BEAM_A", "BEAM_B"],
        "DIAGNOSE": "BEAM_B",
        "REMEDIATE": "BEAM_A",
        "ASSESS_MASTERY": "BEAM_C",
    },
    "frame_element": {
        "MODEL": "FRAME_A",
        "GUIDED_PRACTICE": ["FRAME_A", "FRAME_B"],
        "DIAGNOSE": "FRAME_B",
        "REMEDIATE": "FRAME_A",
        "ASSESS_MASTERY": "FRAME_C",
    },
    "frame_transformation": {
        "MODEL": "FT_A",
        "GUIDED_PRACTICE": ["FT_A", "FT_B"],
        "DIAGNOSE": "FT_B",
        "REMEDIATE": "FT_A",
        "ASSESS_MASTERY": "FT_C",
    },
    "frame_global_stiffness": {
        "MODEL": "FGS_A",
        "GUIDED_PRACTICE": ["FGS_A", "FGS_B"],
        "DIAGNOSE": "FGS_B",
        "REMEDIATE": "FGS_A",
        "ASSESS_MASTERY": "FGS_C",
    },
    "frame_assembly_solution": {
        "MODEL": "FAS_A",
        "GUIDED_PRACTICE": ["FAS_A", "FAS_B"],
        "DIAGNOSE": "FAS_B",
        "REMEDIATE": "FAS_A",
        "ASSESS_MASTERY": "FAS_C",
    },
    "frame_force_recovery": {
        "MODEL": "FFR_A",
        "GUIDED_PRACTICE": ["FFR_A", "FFR_B"],
        "DIAGNOSE": "FFR_B",
        "REMEDIATE": "FFR_A",
        "ASSESS_MASTERY": "FFR_C",
    },
}


# Mapping from curriculum nodes to spring-agent example names.
# Module 1 only — the spring-agent only handles 1D spring-particle systems.
NODE_TO_EXAMPLE = {
    "element_stiffness": "single_spring_k100",
    "local_global_dofs": "three_spring_series",
    "assembly": "three_spring_series",
    "boundary_conditions": "three_spring_series",
    "solution": "nb01_main_example",
    "verification": "nb01_main_example",
}


class Orchestrator:
    """
    Manages the pedagogical state machine, student model, and context
    construction for the teacher and evaluator agents.
    """

    def __init__(self, data_dir: str = "data", student_file: str = None,
                 on_transition=None):
        self.data_dir = Path(data_dir)

        # Load curriculum graph
        with open(self.data_dir / "curriculum_graph.json") as f:
            self.curriculum = json.load(f)

        # Load problem bank
        with open(self.data_dir / "problem_bank.json") as f:
            self.problem_bank = json.load(f)["problems"]

        # Load or initialize student model
        if student_file and Path(student_file).exists():
            with open(student_file) as f:
                self.student = json.load(f)
        else:
            with open(self.data_dir / "student_model_template.json") as f:
                self.student = json.load(f)

        self.student_file = student_file or "data/student_state.json"
        self._pending_evaluation = False
        self._skipped_to_mastery = False

        # Notebook knowledge layer
        self.nb_knowledge = NotebookKnowledge(data_dir)

        # Spring-agent bridge (lazy init)
        self._bridge = None

        # Notebook code executor (lazy init)
        self._executor = None

        # Persistence callback — called after every state transition
        self._on_transition = on_transition

    @property
    def state(self) -> str:
        return self.student["current_state"]

    @state.setter
    def state(self, new_state: str):
        assert new_state in STATES, f"Invalid state: {new_state}"
        self.student["current_state"] = new_state

    @property
    def current_node(self) -> str:
        return self.student["current_node"]

    @property
    def node_data(self) -> dict:
        return self.curriculum["nodes"][self.current_node]

    @property
    def student_node(self) -> dict:
        return self.student["nodes"][self.current_node]

    @property
    def scaffolding_level(self) -> int:
        return self.student["scaffolding_level"]

    @scaffolding_level.setter
    def scaffolding_level(self, level: int):
        self.student["scaffolding_level"] = max(0, min(3, level))

    @property
    def solver_bridge(self):
        """Lazy-init the spring-agent bridge. Returns None if unavailable."""
        if self._bridge is None:
            try:
                from core.spring_agent_bridge import SpringAgentBridge
                self._bridge = SpringAgentBridge()
            except (FileNotFoundError, ImportError):
                pass  # spring-agent not available — tutor works without it
        return self._bridge

    @property
    def code_executor(self) -> NotebookExecutor:
        """Lazy-init the notebook code executor."""
        if self._executor is None:
            self._executor = NotebookExecutor()
        return self._executor

    def needs_evaluation(self) -> bool:
        """Whether the current state requires calling the evaluator."""
        return self.state in ["GUIDED_PRACTICE", "DIAGNOSE", "ASSESS_MASTERY", "ASSESS_PRIOR"]

    def get_current_problem(self) -> dict | None:
        """Get the problem to use in the current state."""
        node = self.current_node
        state = self.state

        if state not in PROBLEM_MAP.get(node, {}):
            return None

        problem_ref = PROBLEM_MAP[node][state]

        if isinstance(problem_ref, list):
            # Rotate through problems based on attempt count
            idx = self.student_node["attempts"] % len(problem_ref)
            problem_id = problem_ref[idx]
        else:
            problem_id = problem_ref

        return self.problem_bank.get(problem_id)

    def build_teacher_context(self) -> dict:
        """
        Build the context dict that the teacher agent needs.
        This includes the current state, CAM technique, topic info,
        scaffolding level, student summary, and relevant problem content.
        """
        node_info = self.node_data
        student_info = self.student_node
        problem = self.get_current_problem()

        # Map state to CAM technique
        cam_technique = {
            "ASSESS_PRIOR": "assessment",
            "MODEL": "modeling",
            "GUIDED_PRACTICE": "coaching",
            "DIAGNOSE": "articulation",
            "REMEDIATE": "remediation",
            "ASSESS_MASTERY": "exploration",
            "ADVANCE": "transition",
        }[self.state]

        context = {
            "state": self.state,
            "cam_technique": cam_technique,
            "node_id": self.current_node,
            "node_title": node_info["title"],
            "learning_objectives": node_info["learning_objectives"],
            "misconceptions": node_info["misconceptions"],
            "physical_reasoning_markers": node_info["physical_reasoning_markers"],
            "scaffolding_level": self.scaffolding_level,
            "student_summary": {
                "p_mastery": student_info["p_mastery"],
                "attempts": student_info["attempts"],
                "misconceptions_observed": student_info["misconceptions_observed"],
                "misconceptions_resolved": student_info["misconceptions_resolved"],
                "lo_status": student_info["lo_status"],
            },
            "transparency": True,  # Agent should be transparent about assessment
        }

        # Notebook knowledge layer
        context["notebook_references"] = self.nb_knowledge.get_references_for_node(
            self.current_node
        )
        context["curriculum_difference"] = self.nb_knowledge.get_curriculum_difference_note(
            self.current_node
        )

        # When advancing, include the next node's title for the transition prompt
        if self.state == "ADVANCE":
            context["evolution_table"] = self.nb_knowledge.get_evolution_table()
            for source, target in self.curriculum["edges"]:
                if source == self.current_node:
                    context["next_node_title"] = self.curriculum["nodes"][target]["title"]
                    break

        if problem:
            context["problem"] = {
                "problem_id": problem["problem_id"],
                "statement": problem["statement"],
                "solution_steps": problem["solution_steps"],
                "final_answer": problem["final_answer"],
            }

            # Add expert narration for MODEL state
            if self.state == "MODEL":
                context["problem"]["expert_narration"] = problem["expert_narration"]

                # If re-modeling due to a persistent misconception, tell the
                # teacher to emphasize the specific gap in the demonstration
                remodel_misc = student_info.pop("_remodel_misconception", None)
                if remodel_misc:
                    misc_desc = node_info["misconceptions"].get(remodel_misc, remodel_misc)
                    context["remodel_misconception"] = {
                        "id": remodel_misc,
                        "description": misc_desc,
                    }

            # Add scaffolded version for GUIDED_PRACTICE
            if self.state == "GUIDED_PRACTICE":
                level_key = f"level_{self.scaffolding_level}"
                scaffolded = problem["scaffolded_versions"].get(level_key)
                if scaffolded:
                    context["problem"]["scaffolded_prompt"] = scaffolded

            # Add evaluation markers for DIAGNOSE and ASSESS states
            if self.state in ["DIAGNOSE", "ASSESS_MASTERY"]:
                context["problem"]["evaluation_markers"] = problem.get("evaluation_markers", {})

        # When modeling, include the upcoming practice problem so the teacher
        # can present it at the end of the demonstration
        if self.state == "MODEL":
            gp_ref = PROBLEM_MAP.get(self.current_node, {}).get("GUIDED_PRACTICE")
            if gp_ref:
                gp_id = gp_ref[self.student_node["attempts"] % len(gp_ref)] if isinstance(gp_ref, list) else gp_ref
                gp_problem = self.problem_bank.get(gp_id)
                if gp_problem:
                    level_key = f"level_{self.scaffolding_level}"
                    context["practice_problem"] = {
                        "statement": gp_problem["statement"],
                        "scaffolded_prompt": gp_problem["scaffolded_versions"].get(level_key, ""),
                    }

        # Spring-agent solver demo (Module 1 only)
        bridge = self.solver_bridge
        if bridge and self.state == "MODEL":
            example_name = NODE_TO_EXAMPLE.get(self.current_node)
            if example_name:
                solver_result = bridge.solve_example(example_name)
                if solver_result.get("status") == "success":
                    context["solver_demo"] = solver_result

        # Notebook code executor demos (all modules)
        if self.state == "MODEL":
            demos = self.code_executor.get_demos_for_node(self.current_node)
            if demos:
                module_map = {
                    "module1_springs": [
                        "element_stiffness", "local_global_dofs", "assembly",
                        "boundary_conditions", "solution", "verification",
                    ],
                    "module2_trusses": [
                        "bar_element", "two_d_dofs", "coordinate_transformation",
                        "global_element_stiffness", "truss_assembly_solution",
                        "truss_force_recovery",
                    ],
                    "module3_frames": [
                        "beam_element", "frame_element", "frame_transformation",
                        "frame_global_stiffness", "frame_assembly_solution",
                        "frame_force_recovery",
                    ],
                }
                for mod, nodes in module_map.items():
                    if self.current_node in nodes:
                        self.code_executor.set_module(mod)
                        break

                demo_results = self.code_executor.run_demo(demos[0])
                if demo_results and all(s["result"]["success"] for s in demo_results):
                    context["code_demo"] = {
                        "demo_name": demos[0],
                        "steps": demo_results,
                    }

        return context

    def build_evaluator_context(self, student_message: str) -> dict:
        """
        Build the context dict for the evaluator agent.
        """
        node_info = self.node_data
        problem = self.get_current_problem()

        context = {
            "student_response": student_message,
            "node_id": self.current_node,
            "learning_objectives": node_info["learning_objectives"],
            "misconceptions": node_info["misconceptions"],
            "physical_reasoning_markers": node_info["physical_reasoning_markers"],
        }

        if problem:
            context["problem"] = {
                "problem_id": problem["problem_id"],
                "solution_steps": problem["solution_steps"],
                "final_answer": problem["final_answer"],
                "evaluation_markers": problem.get("evaluation_markers", {}),
                "targets_los": problem["targets_los"],
            }

        return context

    def process_evaluation(self, evaluation: dict) -> None:
        """
        Process the evaluator's classification and update student model.
        Then determine the state transition.

        Args:
            evaluation: Dict with keys: procedural, conceptual,
                       misconception_id, confidence, physical_reasoning_detected
        """
        student_node = self.student_node
        procedural = evaluation.get("procedural", "incomplete")
        conceptual = evaluation.get("conceptual", "absent")
        misconception_id = evaluation.get("misconception_id")
        confidence = evaluation.get("confidence", "low")
        physical_reasoning = evaluation.get("physical_reasoning_detected", False)

        # Update misconception tracking
        if misconception_id and misconception_id not in student_node["misconceptions_observed"]:
            student_node["misconceptions_observed"].append(misconception_id)

        # Update LO status based on evaluation
        problem = self.get_current_problem()
        if problem:
            for lo in problem.get("targets_los", []):
                if lo in student_node["lo_status"]:
                    if procedural == "correct" and conceptual == "deep":
                        student_node["lo_status"][lo] = "demonstrated"
                    elif procedural == "correct" and conceptual == "mechanical":
                        student_node["lo_status"][lo] = "attempted_mechanical"
                    elif procedural in ["major_error", "incomplete"]:
                        student_node["lo_status"][lo] = "attempted_failed"

        # Resolve misconceptions: if the student demonstrates deep physical
        # reasoning on a problem that targets a previously observed misconception
        # (and the evaluator does NOT flag the misconception again), that
        # misconception is resolved.
        if (problem and procedural == "correct" and conceptual == "deep"
                and physical_reasoning and not misconception_id):
            for m_id in problem.get("targets_misconceptions", []):
                if (m_id in student_node["misconceptions_observed"]
                        and m_id not in student_node["misconceptions_resolved"]):
                    student_node["misconceptions_resolved"].append(m_id)
                    # Reset remediation counter for this misconception
                    rem = student_node.get("remediation_attempts", {})
                    if m_id in rem:
                        del rem[m_id]

        # Update BKT mastery estimate
        observed_correct = (procedural == "correct" and
                            conceptual in ["deep", "mechanical"])
        student_node["p_mastery"] = update_mastery(
            student_node["p_mastery"],
            observed_correct,
            self.student["bkt_params"]
        )

        # Record scaffolding history
        student_node["scaffolding_history"].append(self.scaffolding_level)
        student_node["attempts"] += 1

        # --- State transition logic (conservative policy) ---
        self._transition(evaluation)

        # Auto-save callback
        if self._on_transition:
            self._on_transition()

    def _transition(self, evaluation: dict) -> None:
        """Determine the next state based on evaluation results."""
        procedural = evaluation.get("procedural", "incomplete")
        conceptual = evaluation.get("conceptual", "absent")
        misconception_id = evaluation.get("misconception_id")
        confidence = evaluation.get("confidence", "low")
        physical_reasoning = evaluation.get("physical_reasoning_detected", False)
        student_node = self.student_node

        current = self.state

        if current == "ASSESS_PRIOR":
            if procedural == "correct" and conceptual == "deep":
                # Student might already know this — jump to mastery check
                self.state = "ASSESS_MASTERY"
                self.scaffolding_level = 0
            elif procedural == "correct" and conceptual == "mechanical":
                # Knows the procedure but not the reasoning — model with emphasis on why
                self.state = "MODEL"
            elif procedural in ["minor_error", "incomplete"]:
                # Partial knowledge — full modeling
                self.state = "MODEL"
            else:
                # No prior knowledge — full modeling
                self.state = "MODEL"

        elif current == "GUIDED_PRACTICE":
            if misconception_id:
                # Misconception detected — remediate regardless of correctness
                self.state = "REMEDIATE"
            elif procedural == "correct" and conceptual == "deep":
                # Good performance — check if we can reduce scaffolding
                if self.scaffolding_level > 1:
                    # Fade scaffolding
                    self.scaffolding_level -= 1
                    self.state = "GUIDED_PRACTICE"
                elif self.scaffolding_level == 1:
                    # Ready for unscaffolded assessment
                    self.scaffolding_level = 0
                    self.state = "ASSESS_MASTERY"
                else:
                    self.state = "ASSESS_MASTERY"
            elif procedural == "correct" and conceptual == "mechanical":
                # Right answer, wrong depth — trigger articulation
                self.state = "DIAGNOSE"
            elif procedural == "minor_error" and conceptual == "deep":
                # Good understanding, small mistake — try again same level
                self.state = "GUIDED_PRACTICE"
            elif procedural in ["major_error", "incomplete"]:
                # Major error or incomplete — increase scaffolding
                if self.scaffolding_level < 3:
                    self.scaffolding_level += 1
                    self.state = "GUIDED_PRACTICE"
                else:
                    # Already at max scaffolding — go back to modeling
                    self.state = "MODEL"
            elif conceptual == "absent":
                # Correct answer but no reasoning provided — must probe
                self.state = "DIAGNOSE"
            else:
                # Major error or confusion — increase scaffolding
                if self.scaffolding_level < 3:
                    self.scaffolding_level += 1
                    self.state = "GUIDED_PRACTICE"
                else:
                    # Already at max scaffolding — go back to modeling
                    self.state = "MODEL"

        elif current == "DIAGNOSE":
            if conceptual == "deep" and physical_reasoning:
                # Articulation successful — reset diagnose counter
                student_node["diagnose_count"] = 0
                if self.scaffolding_level <= 1:
                    self.state = "ASSESS_MASTERY"
                    self.scaffolding_level = 0
                else:
                    self.scaffolding_level -= 1
                    self.state = "GUIDED_PRACTICE"
            elif misconception_id:
                student_node["diagnose_count"] = 0
                self.state = "REMEDIATE"
            else:
                # Still mechanical or absent — track consecutive failures
                student_node["diagnose_count"] = student_node.get("diagnose_count", 0) + 1
                if student_node["diagnose_count"] >= 3:
                    # Loop breaker: 3 DIAGNOSE visits without deep reasoning.
                    # The student needs to see the expert reasoning again.
                    student_node["diagnose_count"] = 0
                    self.state = "MODEL"
                else:
                    self.state = "GUIDED_PRACTICE"

        elif current == "REMEDIATE":
            # Track remediation attempts per misconception
            # (the misconception that triggered REMEDIATE is the last observed one)
            last_misc = (student_node["misconceptions_observed"][-1]
                         if student_node["misconceptions_observed"] else None)
            if last_misc:
                rem_attempts = student_node.get("remediation_attempts", {})
                rem_attempts[last_misc] = rem_attempts.get(last_misc, 0) + 1
                student_node["remediation_attempts"] = rem_attempts

                if rem_attempts[last_misc] >= 2:
                    # Loop breaker: 2 remediation attempts for the same
                    # misconception without resolution. Escalate to MODEL
                    # with emphasis on the specific misconception.
                    student_node["_remodel_misconception"] = last_misc
                    self.state = "MODEL"
                else:
                    self.state = "GUIDED_PRACTICE"
            else:
                self.state = "GUIDED_PRACTICE"

        elif current == "ASSESS_MASTERY":
            # Capture and clear the skip flag before any branching
            cold_assessment = self._skipped_to_mastery
            self._skipped_to_mastery = False

            # Conservative: need correct + deep + physical reasoning + high confidence
            unresolved = [m for m in student_node["misconceptions_observed"]
                          if m not in student_node["misconceptions_resolved"]]

            if (procedural == "correct" and conceptual == "deep"
                    and physical_reasoning and confidence == "high"):
                student_node["level0_passed"] = True

                # Mastery override: a perfect cold Level-0 pass (via /skip)
                # bypasses the BKT threshold. This is the strongest possible
                # evidence — the student demonstrated deep physical reasoning
                # on an unseen problem with zero scaffolding and no prior
                # coaching. Unresolved misconceptions still block advancement.
                if cold_assessment and len(unresolved) == 0:
                    student_node["p_mastery"] = max(
                        student_node["p_mastery"],
                        self.student["bkt_params"]["mastery_threshold"]
                    )
                    self.state = "ADVANCE"
                elif check_mastery(student_node["p_mastery"],
                                 self.student["bkt_params"]["mastery_threshold"],
                                 student_node["level0_passed"],
                                 unresolved):
                    self.state = "ADVANCE"
                else:
                    # High quality response but BKT not yet at threshold
                    # Give another practice problem
                    self.scaffolding_level = 1
                    self.state = "GUIDED_PRACTICE"
            elif misconception_id:
                self.state = "REMEDIATE"
            else:
                # Failed mastery assessment — back to practice with scaffolding
                self.scaffolding_level = 2
                self.state = "GUIDED_PRACTICE"

    def advance_to_next_node(self) -> bool:
        """
        Move to the next node in the curriculum graph.
        Returns True if there is a next node, False if curriculum is complete.
        """
        edges = self.curriculum["edges"]
        current = self.current_node

        # Find the next node
        for source, target in edges:
            if source == current:
                self.student["current_node"] = target
                self.state = "ASSESS_PRIOR"
                self.scaffolding_level = 3
                return True

        # No more nodes — curriculum complete
        return False

    def handle_student_override(self, intent: str) -> None:
        """
        Handle cases where the student goes off-script.

        Args:
            intent: One of 'skip', 'repeat', 'ask_question', 'quit'
        """
        if intent == "skip":
            # Student wants to skip ahead — go to mastery check.
            # Flag this as a cold assessment so _transition can apply
            # the mastery override if all four criteria are met.
            self.state = "ASSESS_MASTERY"
            self.scaffolding_level = 0
            self._skipped_to_mastery = True
        elif intent == "repeat":
            # Student wants to see the concept again
            self.state = "MODEL"
        elif intent == "ask_question":
            # Let the teacher handle the question, then return to current state
            pass  # No state change

    def verify_student_code(self, student_code: str, expected: dict) -> dict:
        """Run student code through the executor and check results.

        Parameters
        ----------
        student_code : str
            Student's Python code.
        expected : dict
            Mapping of expression → expected value.

        Returns
        -------
        dict with: success, all_correct, checks
        """
        module_map = {
            "module1_springs": [
                "element_stiffness", "local_global_dofs", "assembly",
                "boundary_conditions", "solution", "verification",
            ],
            "module2_trusses": [
                "bar_element", "two_d_dofs", "coordinate_transformation",
                "global_element_stiffness", "truss_assembly_solution",
                "truss_force_recovery",
            ],
            "module3_frames": [
                "beam_element", "frame_element", "frame_transformation",
                "frame_global_stiffness", "frame_assembly_solution",
                "frame_force_recovery",
            ],
        }
        for mod, nodes in module_map.items():
            if self.current_node in nodes:
                self.code_executor.set_module(mod)
                break

        return self.code_executor.verify_output(student_code, expected)

    def save_student_model(self) -> None:
        """Persist the student model to disk."""
        with open(self.student_file, "w") as f:
            json.dump(self.student, f, indent=2)

    def handle_exploration_turn(self, student_message: str) -> dict:
        """Process a student message during EXPLORATION mode.

        The student interacts with the solver via natural language.
        The teacher translates intent to tool calls and coaches.
        Returns available capabilities for the teacher context.
        """
        bridge = self.solver_bridge
        if bridge is None:
            return {"mode": "exploration", "available": False}
        return {
            "mode": "exploration",
            "available": True,
            "available_tools": bridge.get_available_tools(),
            "has_model": bridge.has_model,
            "has_results": bridge.has_results,
        }

    def get_status_summary(self) -> str:
        """Human-readable summary of current status for debugging."""
        sn = self.student_node
        unresolved = [m for m in sn["misconceptions_observed"]
                      if m not in sn["misconceptions_resolved"]]
        return (
            f"Node: {self.current_node} | State: {self.state} | "
            f"Scaffolding: {self.scaffolding_level} | "
            f"P(mastery): {sn['p_mastery']:.3f} | "
            f"Attempts: {sn['attempts']} | "
            f"Misconceptions: {unresolved} | "
            f"LO Status: {sn['lo_status']}"
        )
