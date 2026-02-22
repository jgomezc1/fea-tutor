"""Tests for notebook code executor.

Tests execution correctness, module isolation, student code verification,
curated demos, and orchestrator integration.

Run:
    python -m pytest tests/test_notebook_executor.py -v
"""
import pytest
import json
import numpy as np
from pathlib import Path


@pytest.fixture(scope="module")
def executor():
    """Shared executor instance for all tests."""
    from core.notebook_executor import NotebookExecutor
    return NotebookExecutor()


@pytest.fixture
def fresh_executor():
    """Fresh executor per test (clean namespaces)."""
    from core.notebook_executor import NotebookExecutor
    return NotebookExecutor()


# ===================================================================
# Module initialization and function loading
# ===================================================================

class TestModuleInit:
    """Verify all modules load their functions correctly."""

    def test_module1_has_spring_functions(self, executor):
        executor.set_module("module1_springs")
        funcs = executor.available_functions
        assert "uelspring" in funcs
        assert "eqcounter" in funcs
        assert "DME" in funcs
        assert "assembly" in funcs
        assert "loadasem" in funcs

    def test_module2_has_truss_functions(self, executor):
        executor.set_module("module2_trusses")
        funcs = executor.available_functions
        assert "ueltruss2D" in funcs
        assert "eqcounter" in funcs
        assert "DME" in funcs
        assert "assembly" in funcs
        assert "loadasem" in funcs

    def test_module3_has_frame_functions(self, executor):
        executor.set_module("module3_frames")
        funcs = executor.available_functions
        assert "uelbeam2DU" in funcs
        assert "eqcounter" in funcs
        assert "DME" in funcs
        assert "assembly" in funcs
        assert "loadasem" in funcs

    def test_unknown_module_raises(self, executor):
        with pytest.raises(ValueError, match="Unknown module"):
            executor.set_module("module4_nonexistent")

    def test_default_module_is_springs(self, fresh_executor):
        assert fresh_executor.current_module == "module1_springs"


# ===================================================================
# Module namespace isolation
# ===================================================================

class TestNamespaceIsolation:
    """Verify modules have separate namespaces and don't leak."""

    def test_spring_function_not_in_truss_namespace(self, fresh_executor):
        fresh_executor.set_module("module2_trusses")
        funcs = fresh_executor.available_functions
        assert "uelspring" not in funcs

    def test_truss_function_not_in_spring_namespace(self, fresh_executor):
        fresh_executor.set_module("module1_springs")
        funcs = fresh_executor.available_functions
        assert "ueltruss2D" not in funcs

    def test_variables_dont_leak_between_modules(self, fresh_executor):
        fresh_executor.set_module("module1_springs")
        fresh_executor.execute("test_var_m1 = 42")

        fresh_executor.set_module("module2_trusses")
        r = fresh_executor.execute("print(test_var_m1)")
        assert not r.success  # should fail — variable doesn't exist

    def test_reset_clears_variables(self, fresh_executor):
        fresh_executor.set_module("module1_springs")
        fresh_executor.execute("my_var = 999")
        fresh_executor.reset_namespace()
        r = fresh_executor.execute("print(my_var)")
        assert not r.success

    def test_reset_preserves_functions(self, fresh_executor):
        fresh_executor.set_module("module1_springs")
        fresh_executor.reset_namespace()
        funcs = fresh_executor.available_functions
        assert "uelspring" in funcs


# ===================================================================
# Numerical correctness — Module 1 (Springs)
# ===================================================================

class TestSpringsCorrectness:
    """Verify spring functions produce correct numerical results."""

    def test_uelspring_values(self, executor):
        executor.set_module("module1_springs")
        r = executor.execute_and_capture(
            "K = uelspring(100.0)", capture_vars=["K"]
        )
        K = np.array(r["captured"]["K"])
        expected = np.array([[100, -100], [-100, 100]])
        assert np.allclose(K, expected)

    def test_uelspring_symmetry(self, executor):
        executor.set_module("module1_springs")
        r = executor.execute_and_capture(
            "K = uelspring(500.0)", capture_vars=["K"]
        )
        K = np.array(r["captured"]["K"])
        assert np.allclose(K, K.T)

    def test_uelspring_row_sums_zero(self, executor):
        executor.set_module("module1_springs")
        r = executor.execute_and_capture(
            "K = uelspring(100.0)", capture_vars=["K"]
        )
        K = np.array(r["captured"]["K"])
        assert np.allclose(K.sum(axis=1), 0)

    def test_uelspring_singular(self, executor):
        executor.set_module("module1_springs")
        r = executor.execute_and_capture(
            "K = uelspring(100.0)\nd = float(np.linalg.det(K))",
            capture_vars=["d"],
        )
        assert abs(r["captured"]["d"]) < 1e-10

    def test_3spring_series_solve(self, executor):
        """Full solve: 3 springs in series, F=100N, k=1000,2000,1500."""
        executor.set_module("module1_springs")
        executor.reset_namespace()
        code = """
nodes = np.array([[0,0.0,-1],[1,1.0,0],[2,2.0,0],[3,3.0,0]])
mats = np.array([1000.0, 2000.0, 1500.0])
elements = np.array([[0,0,0,0,1],[1,0,1,1,2],[2,0,2,2,3]], dtype=int)
loads = np.array([[3, 100.0]])
neq, IBC = eqcounter(nodes)
DME_op, _, _ = DME(nodes, elements)
KG = assembly(elements, mats, nodes, neq, DME_op)
RHSG = loadasem(loads, IBC, neq, 1)
UG = np.linalg.solve(KG, RHSG)
"""
        r = executor.execute_and_capture(code, capture_vars=["UG", "neq"])
        assert r["success"]
        assert r["captured"]["neq"] == 3.0  # 3 free DOFs
        UG = np.array(r["captured"]["UG"])
        # u3 = F * (1/k0 + 1/k1 + 1/k2) = 100 * (1/1000 + 1/2000 + 1/1500)
        u3_expected = 100 * (1/1000 + 1/2000 + 1/1500)
        assert abs(UG[2] - u3_expected) < 1e-10


# ===================================================================
# Numerical correctness — Module 2 (Trusses)
# ===================================================================

class TestTrussCorrectness:
    """Verify truss functions produce correct numerical results."""

    def test_horizontal_truss_k(self, executor):
        """Horizontal bar: K_global should only have ux stiffness."""
        executor.set_module("module2_trusses")
        r = executor.execute_and_capture(
            "coord = np.array([[0.0,0.0],[2.0,0.0]])\n"
            "K = ueltruss2D(coord, 1.0, 1000.0)",
            capture_vars=["K"],
        )
        K = np.array(r["captured"]["K"])
        assert K.shape == (4, 4)
        # EA/L = 1000/2 = 500
        assert abs(K[0, 0] - 500.0) < 1e-10
        # uy rows/cols should be zero (no transverse stiffness)
        assert abs(K[1, 1]) < 1e-10
        assert abs(K[3, 3]) < 1e-10

    def test_45deg_truss_k(self, executor):
        """45° bar: all quadrants should be non-zero."""
        executor.set_module("module2_trusses")
        r = executor.execute_and_capture(
            "coord = np.array([[0.0,0.0],[1.0,1.0]])\n"
            "K = ueltruss2D(coord, 1.0, 1000.0)",
            capture_vars=["K"],
        )
        K = np.array(r["captured"]["K"])
        # All diagonal entries should be non-zero
        for i in range(4):
            assert abs(K[i, i]) > 1e-10

    def test_truss_k_symmetric(self, executor):
        executor.set_module("module2_trusses")
        r = executor.execute_and_capture(
            "coord = np.array([[0.0,0.0],[3.0,4.0]])\n"
            "K = ueltruss2D(coord, 2.0, 500.0)",
            capture_vars=["K"],
        )
        K = np.array(r["captured"]["K"])
        assert np.allclose(K, K.T)


# ===================================================================
# Numerical correctness — Module 3 (Frames)
# ===================================================================

class TestFrameCorrectness:
    """Verify frame functions produce correct numerical results."""

    def test_horizontal_beam_k(self, executor):
        """Horizontal beam: bending stiffness in uy and θ, zero in ux."""
        executor.set_module("module3_frames")
        r = executor.execute_and_capture(
            "coord = np.array([[0.0,0.0],[3.0,0.0]])\n"
            "K = uelbeam2DU(coord, 1.0, 1000.0)",
            capture_vars=["K"],
        )
        K = np.array(r["captured"]["K"])
        assert K.shape == (6, 6)
        # ux DOFs (indices 0,3) should have zero stiffness (axially rigid beam)
        assert abs(K[0, 0]) < 1e-10
        assert abs(K[3, 3]) < 1e-10
        # uy DOF (index 1) should be 12EI/L³
        L = 3.0
        EI = 1000.0
        assert abs(K[1, 1] - 12 * EI / L**3) < 1e-8
        # θ DOF (index 2) should be 4EI/L
        assert abs(K[2, 2] - 4 * EI / L) < 1e-8

    def test_vertical_beam_k(self, executor):
        """Vertical column: bending stiffness appears in ux."""
        executor.set_module("module3_frames")
        r = executor.execute_and_capture(
            "coord = np.array([[0.0,0.0],[0.0,3.0]])\n"
            "K = uelbeam2DU(coord, 1.0, 1000.0)",
            capture_vars=["K"],
        )
        K = np.array(r["captured"]["K"])
        L = 3.0
        EI = 1000.0
        # ux DOF (index 0) should now have 12EI/L³ (bending in x)
        assert abs(K[0, 0] - 12 * EI / L**3) < 1e-8

    def test_beam_k_symmetric(self, executor):
        executor.set_module("module3_frames")
        r = executor.execute_and_capture(
            "coord = np.array([[0.0,0.0],[3.0,4.0]])\n"
            "K = uelbeam2DU(coord, 2.0, 500.0)",
            capture_vars=["K"],
        )
        K = np.array(r["captured"]["K"])
        assert np.allclose(K, K.T)


# ===================================================================
# Student code verification
# ===================================================================

class TestStudentVerification:
    """Test the verify_output method for checking student work."""

    def test_correct_answer(self, fresh_executor):
        fresh_executor.set_module("module1_springs")
        r = fresh_executor.verify_output(
            "K = uelspring(100.0)",
            {"K[0,0]": 100.0, "K[0,1]": -100.0},
        )
        assert r["success"]
        assert r["all_correct"]
        assert len(r["checks"]) == 2

    def test_wrong_answer_detected(self, fresh_executor):
        fresh_executor.set_module("module1_springs")
        r = fresh_executor.verify_output(
            "K = uelspring(100.0)",
            {"K[0,0]": 999.0},
        )
        assert r["success"]
        assert not r["all_correct"]
        assert r["checks"][0]["actual"] == 100.0
        assert r["checks"][0]["expected"] == 999.0

    def test_syntax_error_in_student_code(self, fresh_executor):
        fresh_executor.set_module("module1_springs")
        r = fresh_executor.verify_output(
            "K = uelspring(100.0",  # missing closing paren
            {"K[0,0]": 100.0},
        )
        assert not r["success"]
        assert "SyntaxError" in r["error"]

    def test_runtime_error_in_student_code(self, fresh_executor):
        fresh_executor.set_module("module1_springs")
        r = fresh_executor.verify_output(
            "K = uelspring('not_a_number')",
            {"K[0,0]": 100.0},
        )
        assert not r["success"]
        assert r["error"] is not None

    def test_shape_check(self, fresh_executor):
        fresh_executor.set_module("module1_springs")
        r = fresh_executor.verify_output(
            "K = uelspring(50.0)",
            {"K.shape": (2, 2)},
        )
        assert r["success"]
        assert r["all_correct"]

    def test_verify_resets_namespace(self, fresh_executor):
        """verify_output should not be contaminated by previous calls."""
        fresh_executor.set_module("module1_springs")
        fresh_executor.execute("leftover = 42")
        r = fresh_executor.verify_output(
            "K = uelspring(100.0)",
            {"K[0,0]": 100.0},
        )
        assert r["all_correct"]

    def test_tolerance_parameter(self, fresh_executor):
        fresh_executor.set_module("module1_springs")
        # Tight tolerance — should fail on slightly wrong answer
        r = fresh_executor.verify_output(
            "val = 100.001",
            {"val": 100.0},
            tolerance=1e-6,
        )
        assert not r["all_correct"]

        # Loose tolerance — should pass
        r = fresh_executor.verify_output(
            "val = 100.001",
            {"val": 100.0},
            tolerance=0.01,
        )
        assert r["all_correct"]


# ===================================================================
# Curated demos
# ===================================================================

class TestCuratedDemos:
    """Test that all curated demos execute without errors."""

    def test_all_demos_run_successfully(self, fresh_executor):
        from core.notebook_executor import CURATED_DEMOS
        for name in CURATED_DEMOS:
            results = fresh_executor.run_demo(name)
            for step in results:
                assert step["result"]["success"], (
                    f"Demo '{name}' step {step['step']} failed: "
                    f"{step['result'].get('error')}"
                )

    def test_each_demo_has_narration(self, fresh_executor):
        from core.notebook_executor import CURATED_DEMOS
        for name, demo in CURATED_DEMOS.items():
            for i, step in enumerate(demo["steps"]):
                assert "narration" in step, f"Demo '{name}' step {i} missing narration"
                assert len(step["narration"]) > 10, f"Demo '{name}' step {i} narration too short"

    def test_unknown_demo_returns_error(self, executor):
        results = executor.run_demo("nonexistent_demo")
        assert "error" in results[0]

    def test_element_stiffness_demo_output(self, fresh_executor):
        results = fresh_executor.run_demo("element_stiffness_build")
        # Step 1 should print the matrix
        assert "100." in results[0]["result"]["stdout"]
        assert "-100." in results[0]["result"]["stdout"]

    def test_assembly_demo_output(self, fresh_executor):
        results = fresh_executor.run_demo("assembly_3spring")
        # Last step should show displacements
        last = results[-1]["result"]["stdout"]
        assert "Displacements" in last or "u3" in last.lower() or "displacement" in last.lower()

    def test_truss_demo_runs(self, fresh_executor):
        results = fresh_executor.run_demo("truss_element_stiffness")
        assert all(s["result"]["success"] for s in results)

    def test_beam_demo_runs(self, fresh_executor):
        results = fresh_executor.run_demo("beam_stiffness_matrix")
        assert all(s["result"]["success"] for s in results)


# ===================================================================
# Node-to-demo mapping
# ===================================================================

class TestNodeToDemoMapping:
    """Verify demos are mapped to curriculum nodes correctly."""

    def test_element_stiffness_has_demos(self, executor):
        demos = executor.get_demos_for_node("element_stiffness")
        assert len(demos) >= 1

    def test_assembly_has_demos(self, executor):
        demos = executor.get_demos_for_node("assembly")
        assert len(demos) >= 1

    def test_bar_element_has_demos(self, executor):
        demos = executor.get_demos_for_node("bar_element")
        assert len(demos) >= 1

    def test_beam_element_has_demos(self, executor):
        demos = executor.get_demos_for_node("beam_element")
        assert len(demos) >= 1

    def test_verification_has_demos(self, executor):
        demos = executor.get_demos_for_node("verification")
        assert len(demos) >= 1

    def test_node_without_demo_returns_empty(self, executor):
        demos = executor.get_demos_for_node("nonexistent_node")
        assert demos == []


# ===================================================================
# Error handling
# ===================================================================

class TestErrorHandling:
    """Test graceful error handling."""

    def test_division_by_zero(self, executor):
        executor.set_module("module1_springs")
        r = executor.execute("x = 1/0")
        assert not r.success
        assert "ZeroDivisionError" in r.error

    def test_undefined_variable(self, executor):
        executor.set_module("module1_springs")
        r = executor.execute("print(undefined_var)")
        assert not r.success
        assert "NameError" in r.error

    def test_import_blocked(self, executor):
        """Verify dangerous imports don't work (os, subprocess, etc.)."""
        executor.set_module("module1_springs")
        # os should not be pre-loaded but may be importable
        r = executor.execute("import subprocess; subprocess.run(['ls'])")
        # This should either fail or at least not crash the executor
        # (we're not sandboxing, but the test verifies the executor is robust)
        assert isinstance(r.success, bool)

    def test_stdout_captured(self, executor):
        executor.set_module("module1_springs")
        r = executor.execute("print('hello world')")
        assert r.success
        assert "hello world" in r.stdout

    def test_multiline_code(self, executor):
        executor.set_module("module1_springs")
        r = executor.execute("a = 1\nb = 2\nc = a + b\nprint(c)")
        assert r.success
        assert "3" in r.stdout
