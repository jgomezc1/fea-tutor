"""Bridge to spring-agent solver and tools.

Provides the FEA tutor with access to the spring-particle FEM solver
for demonstrations, verification, and student exploration.

Three usage modes:
  A) Programmatic — tutor creates/solves models for MODEL demos
  B) Verification — check student answers against solver in GUIDED_PRACTICE
  C) Full access  — student drives the solver in EXPLORATION mode

Usage:
    bridge = SpringAgentBridge(spring_agent_path='../spring-agent')

    # Mode A: Create and solve a model
    result = bridge.create_and_solve(model_spec)

    # Mode B: Verify student answer
    check = bridge.verify_displacement(node_id=3, student_answer=0.067)

    # Mode C: Full tool access
    result = bridge.execute('run_sensitivity', {...})
"""
import os
import sys
import copy
from pathlib import Path


class SpringAgentBridge:
    """Bridge between FEA tutor and spring-agent solver/tools."""

    def __init__(self, spring_agent_path: str = None):
        """Initialize bridge by importing spring-agent modules.

        Parameters
        ----------
        spring_agent_path : str or None
            Path to the spring-agent project root directory.
            Must contain solver.py and tools.py.
            If None, reads from SPRING_AGENT_PATH env var (default: ../spring-agent).
        """
        if spring_agent_path is None:
            spring_agent_path = os.environ.get("SPRING_AGENT_PATH", "../spring-agent")
        sa_path = Path(spring_agent_path).resolve()
        if not (sa_path / "solver.py").exists():
            raise FileNotFoundError(
                f"spring-agent not found at {sa_path}. "
                f"Expected solver.py and tools.py in that directory."
            )

        # Add to path for import
        sa_str = str(sa_path)
        if sa_str not in sys.path:
            sys.path.insert(0, sa_str)

        # Import spring-agent modules
        import solver as sa_solver
        import tools as sa_tools

        self._solver = sa_solver
        self._tools = sa_tools
        self._execute_tool = sa_tools.execute_tool
        self._state = self._fresh_state()

    @staticmethod
    def _fresh_state() -> dict:
        """Create a clean state dict for the spring-agent tools."""
        return {
            "model": None,
            "results": None,
            "history": [],
            "cascade_result": None,
            "sensitivity_result": None,
        }

    def reset(self):
        """Clear all state (model, results, history)."""
        self._state = self._fresh_state()

    # ------------------------------------------------------------------
    # Mode A: Programmatic tool calls (MODEL state demos)
    # ------------------------------------------------------------------

    def create_and_solve(self, model_spec: dict) -> dict:
        """Create a model from spec and run full analysis.

        Parameters
        ----------
        model_spec : dict
            Keys: name, nodes, materials, elements, constraints, loads.
            Same schema as the create_model tool input.

        Returns
        -------
        dict
            status, model_summary, results (displacements, element_forces,
            reaction_forces, stiffness_matrix, neq, condition_number),
            validation report.
        """
        self.reset()

        # Create
        create_result = self._execute_tool(
            "create_model", model_spec, self._state
        )
        if create_result.get("status") != "success":
            return {
                "status": "error",
                "stage": "create",
                "message": create_result.get("message", "Create failed"),
            }

        # Validate
        val_result = self._execute_tool(
            "validate_model", {}, self._state
        )
        report = val_result.get("report", val_result)
        if not report.get("valid", False):
            return {
                "status": "error",
                "stage": "validate",
                "message": "Validation failed",
                "validation": val_result,
            }

        # Solve
        solve_result = self._execute_tool(
            "run_analysis", {}, self._state
        )
        if solve_result.get("status") != "success":
            return {
                "status": "error",
                "stage": "solve",
                "message": solve_result.get("message", "Solve failed"),
            }

        return {
            "status": "success",
            "model_summary": create_result.get("model_summary"),
            "results": solve_result.get("results", solve_result),
            "validation": val_result,
        }

    def solve_current(self) -> dict:
        """Run analysis on the currently loaded model (after edits)."""
        if self._state["model"] is None:
            return {"status": "error", "message": "No model loaded."}

        val_result = self._execute_tool(
            "validate_model", {}, self._state
        )
        report = val_result.get("report", val_result)
        if not report.get("valid", False):
            return {
                "status": "error",
                "stage": "validate",
                "validation": val_result,
            }

        solve_result = self._execute_tool(
            "run_analysis", {}, self._state
        )
        return solve_result

    def solve_example(self, example_name: str) -> dict:
        """Solve a pre-defined example that matches a notebook/curriculum problem.

        Parameters
        ----------
        example_name : str
            Key from NOTEBOOK_EXAMPLES.

        Returns
        -------
        dict with model_summary and full results.
        """
        if example_name not in NOTEBOOK_EXAMPLES:
            return {
                "status": "error",
                "message": (
                    f"Unknown example: {example_name}. "
                    f"Available: {list(NOTEBOOK_EXAMPLES.keys())}"
                ),
            }
        return self.create_and_solve(NOTEBOOK_EXAMPLES[example_name])

    # ------------------------------------------------------------------
    # Mode B: Answer verification (GUIDED_PRACTICE / ASSESS_MASTERY)
    # ------------------------------------------------------------------

    def verify_displacement(
        self, node_id: int, student_answer: float, tolerance: float = 0.01
    ) -> dict:
        """Check student's displacement answer against solver.

        Parameters
        ----------
        node_id : int
            Node to check.
        student_answer : float
            Student's computed displacement.
        tolerance : float
            Relative tolerance (0.01 = 1%).  For zero expected values,
            used as absolute tolerance.

        Returns
        -------
        dict with: correct, expected, student, error_pct, abs_error
        """
        return self._verify(
            "displacement", node_id, student_answer, tolerance
        )

    def verify_element_force(
        self, element_id: int, student_answer: float, tolerance: float = 0.01
    ) -> dict:
        """Check student's element force answer against solver."""
        return self._verify(
            "element_force", element_id, student_answer, tolerance
        )

    def verify_reaction(
        self, node_id: int, student_answer: float, tolerance: float = 0.01
    ) -> dict:
        """Check student's reaction force answer against solver."""
        return self._verify(
            "reaction", node_id, student_answer, tolerance
        )

    def _verify(
        self, query_type: str, entity_id: int,
        student_answer: float, tolerance: float
    ) -> dict:
        """Generic verification against solver results."""
        if self._state["results"] is None:
            return {
                "status": "error",
                "message": "No results available. Run create_and_solve first.",
            }

        result = self._execute_tool(
            "query_results",
            {"query_type": query_type, "id": entity_id},
            self._state,
        )
        if result.get("status") != "success":
            return result

        # Extract value — key name depends on query_type
        value_keys = {
            "displacement": "displacement",
            "element_force": "force",
            "reaction": "reaction",
        }
        expected = result.get(value_keys.get(query_type, "value"),
                              result.get("value"))
        abs_error = abs(student_answer - expected)

        if abs(expected) < 1e-12:
            correct = abs_error < tolerance
            error_pct = None
        else:
            error_pct = abs_error / abs(expected) * 100
            correct = error_pct < (tolerance * 100)

        return {
            "correct": correct,
            "expected": expected,
            "student": student_answer,
            "error_pct": error_pct,
            "abs_error": abs_error,
        }

    def get_all_results(self) -> dict:
        """Return full result summary for teacher comparison."""
        if self._state["results"] is None:
            return {"status": "error", "message": "No results available."}
        return self._execute_tool(
            "query_results", {"query_type": "summary"}, self._state
        )

    def get_stiffness_matrix(self) -> list | None:
        """Return the assembled (reduced) global stiffness matrix."""
        if self._state["results"] is None:
            return None
        return self._state["results"].get("stiffness_matrix")

    # ------------------------------------------------------------------
    # Mode C: Full tool access (EXPLORATION mode)
    # ------------------------------------------------------------------

    def execute(self, tool_name: str, tool_input: dict | None = None) -> dict:
        """Execute any spring-agent tool directly.

        Parameters
        ----------
        tool_name : str
            One of the 11 spring-agent tools (+ compare_models).
        tool_input : dict
            Tool-specific input parameters.

        Returns
        -------
        dict : Tool-specific result.
        """
        return self._execute_tool(
            tool_name, tool_input or {}, self._state
        )

    def get_available_tools(self) -> list[str]:
        """Return list of available tool names."""
        return [t["name"] for t in self._tools.TOOL_DEFINITIONS]

    def get_tool_definitions(self) -> list[dict]:
        """Return Anthropic tool-use schema definitions.

        Useful for passing to Claude in EXPLORATION mode so the student
        can interact with the solver via natural language.
        """
        return self._tools.TOOL_DEFINITIONS

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def has_model(self) -> bool:
        return self._state["model"] is not None

    @property
    def has_results(self) -> bool:
        return self._state["results"] is not None

    @property
    def current_model(self) -> dict | None:
        return self._state["model"]

    @property
    def current_results(self) -> dict | None:
        return self._state["results"]


# ======================================================================
# Pre-defined examples for tutor demonstrations
# ======================================================================

NOTEBOOK_EXAMPLES = {
    # ----- Module 1: Springs -----

    # Matches nb01 example: 5 nodes, 4 springs, fixed at both ends
    "nb01_main_example": {
        "name": "nb01: 5-node spring system",
        "nodes": [
            {"id": 0, "x": 0.0},
            {"id": 1, "x": 1.0},
            {"id": 2, "x": 2.0},
            {"id": 3, "x": 3.0},
            {"id": 4, "x": 4.0},
        ],
        "materials": [
            {"id": 0, "k": 800.0},
            {"id": 1, "k": 500.0},
            {"id": 2, "k": 1000.0},
        ],
        "elements": [
            {"id": 0, "mat": 0, "conn": [0, 1]},
            {"id": 1, "mat": 0, "conn": [1, 2]},
            {"id": 2, "mat": 1, "conn": [2, 3]},
            {"id": 3, "mat": 2, "conn": [3, 4]},
        ],
        "constraints": [{"node": 0}, {"node": 4}],
        "loads": [
            {"node": 1, "value": 2000.0},
            {"node": 2, "value": 0.0},
            {"node": 3, "value": -1500.0},
        ],
    },

    # Single spring — element_stiffness node (ES_A problem)
    "single_spring_k100": {
        "name": "Single spring k=100 N/m",
        "nodes": [
            {"id": 0, "x": 0.0},
            {"id": 1, "x": 1.0},
        ],
        "materials": [{"id": 0, "k": 100.0}],
        "elements": [{"id": 0, "mat": 0, "conn": [0, 1]}],
        "constraints": [{"node": 0}],
        "loads": [{"node": 1, "value": 50.0}],
    },

    # 3-spring series — assembly and solution demo
    "three_spring_series": {
        "name": "3-spring series system",
        "nodes": [
            {"id": 0, "x": 0.0},
            {"id": 1, "x": 1.0},
            {"id": 2, "x": 2.0},
            {"id": 3, "x": 3.0},
        ],
        "materials": [
            {"id": 0, "k": 1000.0},
            {"id": 1, "k": 2000.0},
            {"id": 2, "k": 1500.0},
        ],
        "elements": [
            {"id": 0, "mat": 0, "conn": [0, 1]},
            {"id": 1, "mat": 1, "conn": [1, 2]},
            {"id": 2, "mat": 2, "conn": [2, 3]},
        ],
        "constraints": [{"node": 0}],
        "loads": [{"node": 3, "value": 100.0}],
    },

    # Parallel springs — conceptual understanding
    "parallel_springs": {
        "name": "Two parallel springs",
        "nodes": [
            {"id": 0, "x": 0.0},
            {"id": 1, "x": 1.0},
        ],
        "materials": [
            {"id": 0, "k": 500.0},
            {"id": 1, "k": 300.0},
        ],
        "elements": [
            {"id": 0, "mat": 0, "conn": [0, 1]},
            {"id": 1, "mat": 1, "conn": [0, 1]},
        ],
        "constraints": [{"node": 0}],
        "loads": [{"node": 1, "value": 200.0}],
    },

    # Branching — node connected to 3 springs
    "branching_system": {
        "name": "Branching: node 1 connected to 3 springs",
        "nodes": [
            {"id": 0, "x": 0.0},
            {"id": 1, "x": 1.0},
            {"id": 2, "x": 2.0},
            {"id": 3, "x": 2.0},
        ],
        "materials": [
            {"id": 0, "k": 1000.0},
            {"id": 1, "k": 800.0},
            {"id": 2, "k": 1200.0},
        ],
        "elements": [
            {"id": 0, "mat": 0, "conn": [0, 1]},
            {"id": 1, "mat": 1, "conn": [1, 2]},
            {"id": 2, "mat": 2, "conn": [1, 3]},
        ],
        "constraints": [{"node": 0}],
        "loads": [{"node": 2, "value": 50.0}, {"node": 3, "value": 75.0}],
    },

    # Failure cascade — 3 springs in series with finite strength
    "failure_cascade_demo": {
        "name": "Failure cascade: 3 springs with f_ult",
        "nodes": [
            {"id": 0, "x": 0.0},
            {"id": 1, "x": 1.0},
            {"id": 2, "x": 2.0},
            {"id": 3, "x": 3.0},
        ],
        "materials": [
            {"id": 0, "k": 1000.0, "f_ult": 80.0},
            {"id": 1, "k": 1000.0, "f_ult": 60.0},
            {"id": 2, "k": 1000.0, "f_ult": 100.0},
        ],
        "elements": [
            {"id": 0, "mat": 0, "conn": [0, 1]},
            {"id": 1, "mat": 1, "conn": [1, 2]},
            {"id": 2, "mat": 2, "conn": [2, 3]},
        ],
        "constraints": [{"node": 0}],
        "loads": [{"node": 3, "value": 100.0}],
    },

    # Sensitivity demo — which spring matters most?
    "sensitivity_demo": {
        "name": "Sensitivity: 4-spring mixed system",
        "nodes": [
            {"id": 0, "x": 0.0},
            {"id": 1, "x": 1.0},
            {"id": 2, "x": 2.0},
        ],
        "materials": [
            {"id": 0, "k": 500.0},
            {"id": 1, "k": 1000.0},
            {"id": 2, "k": 200.0},
        ],
        "elements": [
            {"id": 0, "mat": 0, "conn": [0, 1]},
            {"id": 1, "mat": 1, "conn": [0, 1]},  # parallel with elem 0
            {"id": 2, "mat": 2, "conn": [1, 2]},
        ],
        "constraints": [{"node": 0}],
        "loads": [{"node": 2, "value": 100.0}],
    },
}
