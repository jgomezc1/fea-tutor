"""Tests for spring-agent bridge integration.

These tests validate the bridge module independently by mocking
the spring-agent imports. Tests that require the actual spring-agent
are marked with @pytest.mark.integration.
"""
import pytest
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Unit tests (no spring-agent dependency)
# ---------------------------------------------------------------------------


class TestNotebookExamples:
    """Validate the pre-defined example models are well-formed."""

    def setup_method(self):
        # Import the examples dict directly (no spring-agent needed)
        spec = Path(__file__).parent.parent / "core" / "spring_agent_bridge.py"
        if spec.exists():
            import importlib.util
            s = importlib.util.spec_from_file_location("bridge", spec)
            mod = importlib.util.module_from_spec(s)
            # We can't fully import without spring-agent, so parse examples manually
            pass

        # Fallback: load examples from the module-level dict
        from core.spring_agent_bridge import NOTEBOOK_EXAMPLES
        self.examples = NOTEBOOK_EXAMPLES

    def test_all_examples_have_required_keys(self):
        required = {"name", "nodes", "materials", "elements", "constraints", "loads"}
        for name, spec in self.examples.items():
            assert required.issubset(spec.keys()), (
                f"Example '{name}' missing keys: {required - spec.keys()}"
            )

    def test_all_examples_have_at_least_one_constraint(self):
        for name, spec in self.examples.items():
            assert len(spec["constraints"]) >= 1, (
                f"Example '{name}' has no constraints"
            )

    def test_all_examples_have_at_least_one_load(self):
        for name, spec in self.examples.items():
            assert len(spec["loads"]) >= 1, (
                f"Example '{name}' has no loads"
            )

    def test_all_examples_have_positive_stiffness(self):
        for name, spec in self.examples.items():
            for mat in spec["materials"]:
                assert mat["k"] > 0, (
                    f"Example '{name}' material {mat['id']} has k={mat['k']}"
                )

    def test_all_element_nodes_exist(self):
        for name, spec in self.examples.items():
            node_ids = {n["id"] for n in spec["nodes"]}
            for elem in spec["elements"]:
                for nid in elem["conn"]:
                    assert nid in node_ids, (
                        f"Example '{name}' elem {elem['id']} "
                        f"references missing node {nid}"
                    )

    def test_all_element_materials_exist(self):
        for name, spec in self.examples.items():
            mat_ids = {m["id"] for m in spec["materials"]}
            for elem in spec["elements"]:
                assert elem["mat"] in mat_ids, (
                    f"Example '{name}' elem {elem['id']} "
                    f"references missing material {elem['mat']}"
                )

    def test_constrained_nodes_exist(self):
        for name, spec in self.examples.items():
            node_ids = {n["id"] for n in spec["nodes"]}
            for con in spec["constraints"]:
                assert con["node"] in node_ids, (
                    f"Example '{name}' constraint on missing node {con['node']}"
                )

    def test_loaded_nodes_exist(self):
        for name, spec in self.examples.items():
            node_ids = {n["id"] for n in spec["nodes"]}
            for load in spec["loads"]:
                assert load["node"] in node_ids, (
                    f"Example '{name}' load on missing node {load['node']}"
                )

    def test_no_duplicate_node_ids(self):
        for name, spec in self.examples.items():
            ids = [n["id"] for n in spec["nodes"]]
            assert len(ids) == len(set(ids)), (
                f"Example '{name}' has duplicate node IDs"
            )

    def test_no_duplicate_element_ids(self):
        for name, spec in self.examples.items():
            ids = [e["id"] for e in spec["elements"]]
            assert len(ids) == len(set(ids)), (
                f"Example '{name}' has duplicate element IDs"
            )

    def test_example_count(self):
        assert len(self.examples) >= 7, (
            f"Expected at least 7 examples, got {len(self.examples)}"
        )

    def test_failure_cascade_example_has_f_ult(self):
        spec = self.examples["failure_cascade_demo"]
        has_strength = any(
            m.get("f_ult") is not None for m in spec["materials"]
        )
        assert has_strength, "Cascade example needs materials with f_ult"


# ---------------------------------------------------------------------------
# Integration tests (require spring-agent)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestBridgeWithSolver:
    """Tests that require the actual spring-agent solver."""

    def setup_method(self):
        from core.spring_agent_bridge import SpringAgentBridge
        # Assumes spring-agent is a sibling directory
        self.bridge = SpringAgentBridge()

    def test_create_and_solve_single_spring(self):
        from core.spring_agent_bridge import NOTEBOOK_EXAMPLES
        result = self.bridge.create_and_solve(
            NOTEBOOK_EXAMPLES["single_spring_k100"]
        )
        assert result["status"] == "success"
        # u = F/k = 50/100 = 0.5
        disp = result["results"]["displacements"]
        assert abs(disp[1] - 0.5) < 1e-10

    def test_verify_displacement_correct(self):
        from core.spring_agent_bridge import NOTEBOOK_EXAMPLES
        self.bridge.create_and_solve(
            NOTEBOOK_EXAMPLES["single_spring_k100"]
        )
        check = self.bridge.verify_displacement(1, 0.5)
        assert check["correct"] is True

    def test_verify_displacement_incorrect(self):
        from core.spring_agent_bridge import NOTEBOOK_EXAMPLES
        self.bridge.create_and_solve(
            NOTEBOOK_EXAMPLES["single_spring_k100"]
        )
        check = self.bridge.verify_displacement(1, 0.6)
        assert check["correct"] is False

    def test_solve_example_by_name(self):
        result = self.bridge.solve_example("three_spring_series")
        assert result["status"] == "success"

    def test_solve_example_unknown_name(self):
        result = self.bridge.solve_example("nonexistent")
        assert result["status"] == "error"

    def test_full_tool_execute(self):
        from core.spring_agent_bridge import NOTEBOOK_EXAMPLES
        self.bridge.create_and_solve(
            NOTEBOOK_EXAMPLES["three_spring_series"]
        )
        result = self.bridge.execute(
            "query_results", {"query_type": "all_displacements"}
        )
        assert result["status"] == "success"

    def test_reset_clears_state(self):
        from core.spring_agent_bridge import NOTEBOOK_EXAMPLES
        self.bridge.create_and_solve(
            NOTEBOOK_EXAMPLES["single_spring_k100"]
        )
        assert self.bridge.has_model
        assert self.bridge.has_results
        self.bridge.reset()
        assert not self.bridge.has_model
        assert not self.bridge.has_results

    def test_parallel_springs_displacement(self):
        from core.spring_agent_bridge import NOTEBOOK_EXAMPLES
        result = self.bridge.create_and_solve(
            NOTEBOOK_EXAMPLES["parallel_springs"]
        )
        assert result["status"] == "success"
        # k_eq = 500 + 300 = 800, u = 200/800 = 0.25
        disp = result["results"]["displacements"]
        assert abs(disp[1] - 0.25) < 1e-10

    def test_get_stiffness_matrix(self):
        from core.spring_agent_bridge import NOTEBOOK_EXAMPLES
        self.bridge.create_and_solve(
            NOTEBOOK_EXAMPLES["single_spring_k100"]
        )
        K = self.bridge.get_stiffness_matrix()
        assert K is not None
        assert len(K) == 1  # 1 free DOF
        assert abs(K[0][0] - 100.0) < 1e-10

    def test_sensitivity_demo(self):
        from core.spring_agent_bridge import NOTEBOOK_EXAMPLES
        self.bridge.create_and_solve(
            NOTEBOOK_EXAMPLES["sensitivity_demo"]
        )
        result = self.bridge.execute(
            "run_sensitivity",
            {"target": {"type": "displacement", "id": 2}},
        )
        assert result["status"] == "success"

    def test_failure_cascade_demo(self):
        from core.spring_agent_bridge import NOTEBOOK_EXAMPLES
        self.bridge.create_and_solve(
            NOTEBOOK_EXAMPLES["failure_cascade_demo"]
        )
        result = self.bridge.execute("run_failure_cascade", {})
        assert result["status"] == "success"
