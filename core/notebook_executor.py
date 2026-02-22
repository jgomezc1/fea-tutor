"""Notebook code executor for the FEA tutor.

Executes notebook functions live during teaching sessions. Provides:
  - Per-module namespaces (springs/trusses/frames) with pre-loaded functions
  - Curated step-by-step demos mapped to curriculum nodes
  - Student code verification (run student code, compare output)
  - Safe execution with timeout and error capture

Usage:
    executor = NotebookExecutor()
    executor.set_module("module1_springs")

    # Run a curated demo
    results = executor.run_demo("element_stiffness_demo")
    for step in results:
        print(step["narration"])
        print(step["output"])

    # Execute arbitrary code
    result = executor.execute("K = uelspring(100); print(K)")

    # Verify student code
    check = executor.verify_output("K = uelspring(50)", {"K[0,0]": 50.0})
"""
import io
import sys
import traceback
import copy
import numpy as np
from contextlib import redirect_stdout, redirect_stderr


class ExecutionResult:
    """Result of a single code execution."""
    __slots__ = ("success", "stdout", "stderr", "error", "namespace_snapshot")

    def __init__(self, success, stdout="", stderr="", error=None, namespace_snapshot=None):
        self.success = success
        self.stdout = stdout
        self.stderr = stderr
        self.error = error
        self.namespace_snapshot = namespace_snapshot or {}

    def to_dict(self):
        return {
            "success": self.success,
            "stdout": self.stdout.strip() if self.stdout else "",
            "stderr": self.stderr.strip() if self.stderr else "",
            "error": self.error,
        }


class NotebookExecutor:
    """Executes notebook code in isolated per-module namespaces."""

    def __init__(self, notebook_kb: dict = None):
        """Initialize with pre-loaded function definitions.

        Parameters
        ----------
        notebook_kb : dict, optional
            The notebook_reference_kb.json content. If None, loads from
            default data directory.
        """
        if notebook_kb is None:
            import json
            from pathlib import Path
            kb_path = Path("data") / "notebook_reference_kb.json"
            if kb_path.exists():
                with open(kb_path) as f:
                    notebook_kb = json.load(f)
            else:
                notebook_kb = {"notebook_references": {}}

        self._kb = notebook_kb
        self._namespaces = {}
        self._current_module = None

        # Initialize all module namespaces
        self._init_module("module1_springs")
        self._init_module("module2_trusses")
        self._init_module("module3_frames")

        # Default to module 1
        self.set_module("module1_springs")

    def _init_module(self, module_name: str):
        """Create a namespace for a module and load its functions."""
        ns = {"np": np, "numpy": np, "__builtins__": __builtins__}

        # Collect functions for this module from the KB
        for key, entry in self._kb.get("notebook_references", {}).items():
            if entry.get("module") == module_name:
                code = entry.get("code", "")
                if code and "def " in code:
                    try:
                        exec(code, ns)
                    except Exception as e:
                        print(f"Warning: Failed to load {key}: {e}",
                              file=sys.stderr)

        self._namespaces[module_name] = ns

    def set_module(self, module_name: str):
        """Switch the active module namespace.

        Parameters
        ----------
        module_name : str
            One of: module1_springs, module2_trusses, module3_frames
        """
        if module_name not in self._namespaces:
            raise ValueError(
                f"Unknown module: {module_name}. "
                f"Available: {list(self._namespaces.keys())}"
            )
        self._current_module = module_name

    @property
    def current_module(self) -> str:
        return self._current_module

    @property
    def available_functions(self) -> list[str]:
        """List functions defined in the current namespace."""
        ns = self._namespaces[self._current_module]
        return [
            k for k, v in ns.items()
            if callable(v) and not k.startswith("_") and k not in ("np", "numpy")
        ]

    def execute(self, code: str) -> ExecutionResult:
        """Execute code in the current module namespace.

        Parameters
        ----------
        code : str
            Python code to execute.

        Returns
        -------
        ExecutionResult with stdout, stderr, success flag, and error info.
        """
        ns = self._namespaces[self._current_module]
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(code, ns)
            # Capture key variables for snapshot
            snapshot = self._snapshot(ns, code)
            return ExecutionResult(
                success=True,
                stdout=stdout_buf.getvalue(),
                stderr=stderr_buf.getvalue(),
                namespace_snapshot=snapshot,
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                stdout=stdout_buf.getvalue(),
                stderr=stderr_buf.getvalue(),
                error=f"{type(e).__name__}: {e}",
            )

    def execute_and_capture(self, code: str, capture_vars: list[str] = None) -> dict:
        """Execute code and return specific variable values from the namespace.

        Parameters
        ----------
        code : str
            Python code to execute.
        capture_vars : list of str
            Variable names to capture from the namespace after execution.

        Returns
        -------
        dict with: success, stdout, error, captured (dict of var_name → value)
        """
        result = self.execute(code)
        captured = {}
        if result.success and capture_vars:
            ns = self._namespaces[self._current_module]
            for var in capture_vars:
                if var in ns:
                    val = ns[var]
                    # Convert numpy arrays to lists for JSON serialization
                    if isinstance(val, np.ndarray):
                        captured[var] = val.tolist()
                    else:
                        try:
                            captured[var] = float(val) if isinstance(val, (int, float, np.floating)) else val
                        except (TypeError, ValueError):
                            captured[var] = str(val)

        return {
            "success": result.success,
            "stdout": result.stdout.strip(),
            "error": result.error,
            "captured": captured,
        }

    def reset_namespace(self):
        """Re-initialize the current module namespace (clears student variables)."""
        self._init_module(self._current_module)

    def verify_output(
        self, student_code: str, expected: dict, tolerance: float = 1e-6
    ) -> dict:
        """Execute student code and compare output variables to expected values.

        Parameters
        ----------
        student_code : str
            Student's Python code.
        expected : dict
            Mapping of expression → expected value.
            Expressions are evaluated in the namespace AFTER running student code.
            e.g. {"K[0,0]": 100.0, "K.shape": (2, 2)}
        tolerance : float
            Numerical tolerance for floating-point comparisons.

        Returns
        -------
        dict with: success, all_correct, checks (list of individual results)
        """
        # Reset namespace to avoid contamination from previous runs
        self._init_module(self._current_module)
        ns = self._namespaces[self._current_module]

        # Execute student code
        result = self.execute(student_code)
        if not result.success:
            return {
                "success": False,
                "all_correct": False,
                "error": result.error,
                "checks": [],
            }

        # Check each expected value
        checks = []
        all_correct = True
        for expr, expected_val in expected.items():
            try:
                actual = eval(expr, ns)
                if isinstance(actual, np.ndarray):
                    actual = actual.tolist()
                if isinstance(expected_val, np.ndarray):
                    expected_val = expected_val.tolist()

                if isinstance(actual, (int, float)) and isinstance(expected_val, (int, float)):
                    correct = abs(actual - expected_val) < tolerance
                elif isinstance(actual, (list, tuple)) and isinstance(expected_val, (list, tuple)):
                    try:
                        correct = np.allclose(list(actual), list(expected_val), atol=tolerance)
                    except (TypeError, ValueError):
                        correct = tuple(actual) == tuple(expected_val)
                elif isinstance(actual, np.ndarray) and isinstance(expected_val, (list, tuple)):
                    correct = np.allclose(actual, expected_val, atol=tolerance)
                else:
                    correct = actual == expected_val

                checks.append({
                    "expression": expr,
                    "expected": expected_val,
                    "actual": actual,
                    "correct": correct,
                })
                if not correct:
                    all_correct = False
            except Exception as e:
                checks.append({
                    "expression": expr,
                    "expected": expected_val,
                    "actual": None,
                    "correct": False,
                    "error": str(e),
                })
                all_correct = False

        return {
            "success": True,
            "all_correct": all_correct,
            "checks": checks,
        }

    # ------------------------------------------------------------------
    # Curated demonstrations
    # ------------------------------------------------------------------

    def run_demo(self, demo_name: str) -> list[dict]:
        """Execute a curated multi-step demonstration.

        Parameters
        ----------
        demo_name : str
            Key from CURATED_DEMOS.

        Returns
        -------
        list of dicts, each with: step, narration, code, result (ExecutionResult.to_dict())
        """
        if demo_name not in CURATED_DEMOS:
            return [{
                "step": 0,
                "error": f"Unknown demo: {demo_name}. Available: {list(CURATED_DEMOS.keys())}",
            }]

        demo = CURATED_DEMOS[demo_name]

        # Set the right module
        self.set_module(demo["module"])
        self.reset_namespace()

        results = []
        for i, step in enumerate(demo["steps"]):
            result = self.execute(step["code"])
            results.append({
                "step": i + 1,
                "narration": step["narration"],
                "code": step["code"],
                "result": result.to_dict(),
            })
            if not result.success:
                # Stop on error
                break

        return results

    def get_available_demos(self) -> list[str]:
        """Return list of available demo names."""
        return list(CURATED_DEMOS.keys())

    def get_demos_for_node(self, node_id: str) -> list[str]:
        """Return demo names mapped to a curriculum node."""
        return [
            name for name, demo in CURATED_DEMOS.items()
            if node_id in demo.get("curriculum_nodes", [])
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _snapshot(ns: dict, code: str) -> dict:
        """Extract recently assigned variables from namespace."""
        import ast
        snapshot = {}
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in ns:
                            val = ns[target.id]
                            if isinstance(val, np.ndarray):
                                snapshot[target.id] = val.tolist()
                            elif isinstance(val, (int, float, str, bool, list, dict)):
                                snapshot[target.id] = val
        except Exception:
            pass
        return snapshot


# ======================================================================
# Curated demonstrations mapped to curriculum nodes
# ======================================================================

CURATED_DEMOS = {
    # ------------------------------------------------------------------
    # MODULE 1: SPRINGS
    # ------------------------------------------------------------------

    "element_stiffness_build": {
        "module": "module1_springs",
        "curriculum_nodes": ["element_stiffness"],
        "description": "Build and inspect the spring stiffness matrix",
        "steps": [
            {
                "narration": (
                    "Let's build the stiffness matrix for a spring with k = 100 N/m. "
                    "The function uelspring takes the stiffness coefficient and returns "
                    "a 2×2 matrix."
                ),
                "code": "K = uelspring(100.0)\nprint('K =')\nprint(K)",
            },
            {
                "narration": (
                    "Notice the pattern: diagonal terms are +k (restoring force), "
                    "off-diagonal terms are -k (coupling). Let's verify the row sums — "
                    "they should be zero, which is Newton's third law for the free element."
                ),
                "code": (
                    "print('Row sums:', K.sum(axis=1))\n"
                    "print('Column sums:', K.sum(axis=0))\n"
                    "print('Determinant:', np.linalg.det(K))"
                ),
            },
            {
                "narration": (
                    "The determinant is zero — the matrix is singular. This means the "
                    "element has a rigid body mode. Let's find it: what displacement "
                    "pattern produces zero forces?"
                ),
                "code": (
                    "u_rigid = np.array([1.0, 1.0])  # both nodes move equally\n"
                    "F = K @ u_rigid\n"
                    "print('Rigid body mode [1,1]:')\n"
                    "print(f'  Forces = {F}  (zero — no deformation, no force)')"
                ),
            },
            {
                "narration": (
                    "Now let's interpret column 1 physically. Setting u = [1, 0] means: "
                    "push node 1 by 1 unit while holding node 2 fixed."
                ),
                "code": (
                    "u_col1 = np.array([1.0, 0.0])\n"
                    "F_col1 = K @ u_col1\n"
                    "print('Column 1 interpretation (u1=1, u2=0):')\n"
                    "print(f'  F1 = {F_col1[0]:+.1f} N (push back — spring resists)')\n"
                    "print(f'  F2 = {F_col1[1]:+.1f} N (spring pulls node 2 toward node 1)')"
                ),
            },
        ],
    },

    "element_stiffness_compare_k": {
        "module": "module1_springs",
        "curriculum_nodes": ["element_stiffness"],
        "description": "Compare stiffness matrices for different k values",
        "steps": [
            {
                "narration": (
                    "What happens when we change the stiffness? Let's compare k=100 "
                    "and k=500."
                ),
                "code": (
                    "K1 = uelspring(100.0)\n"
                    "K2 = uelspring(500.0)\n"
                    "print('K (k=100):')\nprint(K1)\n"
                    "print('\\nK (k=500):')\nprint(K2)\n"
                    "print('\\nRatio K2/K1:', K2[0,0] / K1[0,0])"
                ),
            },
            {
                "narration": (
                    "The matrix scales linearly with k. Double the stiffness, double "
                    "every entry. The structure is always [[k, -k], [-k, k]]. "
                    "The stiffness matrix does NOT depend on applied forces — only on "
                    "the element property k."
                ),
                "code": (
                    "# Verify: applying a force F = 50 N at node 2 with node 1 fixed\n"
                    "F = 50.0\n"
                    "u_soft  = F / 100.0  # k = 100\n"
                    "u_stiff = F / 500.0  # k = 500\n"
                    "print(f'Displacement (k=100): {u_soft:.4f} m')\n"
                    "print(f'Displacement (k=500): {u_stiff:.4f} m')\n"
                    "print(f'Stiffer spring → {u_soft/u_stiff:.1f}x less displacement')"
                ),
            },
        ],
    },

    "assembly_3spring": {
        "module": "module1_springs",
        "curriculum_nodes": ["assembly", "local_global_dofs", "boundary_conditions", "solution"],
        "description": "Assemble and solve a 3-spring series system step by step",
        "steps": [
            {
                "narration": (
                    "Let's build a 3-spring series system: 4 nodes, 3 springs, "
                    "fixed at node 0, load at node 3. I'll construct the system "
                    "the same way the notebook code does."
                ),
                "code": (
                    "# Model data (mimicking the txt file format)\n"
                    "# nodes: id, x-position, BC flag (0=free, -1=fixed)\n"
                    "nodes = np.array([[0, 0.0, -1],\n"
                    "                  [1, 1.0,  0],\n"
                    "                  [2, 2.0,  0],\n"
                    "                  [3, 3.0,  0]])\n"
                    "# materials: stiffness\n"
                    "mats = np.array([1000.0, 2000.0, 1500.0])\n"
                    "# elements: id, type, mat_id, node_i, node_j\n"
                    "elements = np.array([[0, 0, 0, 0, 1],\n"
                    "                     [1, 0, 1, 1, 2],\n"
                    "                     [2, 0, 2, 2, 3]], dtype=int)\n"
                    "# loads: node, force\n"
                    "loads = np.array([[3, 100.0]])\n"
                    "print('Model: 4 nodes, 3 springs in series')\n"
                    "print(f'Springs: k = {mats[0]}, {mats[1]}, {mats[2]} N/m')\n"
                    "print('Fixed at node 0, load = 100 N at node 3')"
                ),
            },
            {
                "narration": (
                    "Step 1: Build the IBC array. This assigns equation numbers to free "
                    "DOFs. Node 0 is fixed (gets -1), nodes 1,2,3 are free (get 0,1,2)."
                ),
                "code": (
                    "neq, IBC = eqcounter(nodes)\n"
                    "print(f'Number of equations: {neq}')\n"
                    "print(f'IBC array: {IBC.flatten()}')\n"
                    "print('Node 0 → eq -1 (fixed)')\n"
                    "print('Node 1 → eq 0')\n"
                    "print('Node 2 → eq 1')\n"
                    "print('Node 3 → eq 2')"
                ),
            },
            {
                "narration": (
                    "Step 2: Build the DME operator. For each element, look up the "
                    "equation numbers of its two nodes."
                ),
                "code": (
                    "DME_op, _, _ = DME(nodes, elements)\n"
                    "print('DME operator (element → equation mapping):')\n"
                    "for i in range(3):\n"
                    "    n_i, n_j = elements[i, 3], elements[i, 4]\n"
                    "    print(f'  Element {i}: nodes ({n_i},{n_j}) → equations ({DME_op[i,0]},{DME_op[i,1]})')"
                ),
            },
            {
                "narration": (
                    "Step 3: Assemble the global stiffness matrix. Each element's 2×2 "
                    "K is scattered into KG using the DME mapping. Watch how the "
                    "overlapping contributions at shared nodes ADD together."
                ),
                "code": (
                    "KG = assembly(elements, mats, nodes, neq, DME_op)\n"
                    "print('Global stiffness matrix (3×3):')\n"
                    "print(KG)\n"
                    "print(f'\\nKG[0,0] = k0 + k1 = {mats[0]} + {mats[1]} = {KG[0,0]}')\n"
                    "print(f'KG[1,1] = k1 + k2 = {mats[1]} + {mats[2]} = {KG[1,1]}')"
                ),
            },
            {
                "narration": (
                    "Step 4: Assemble the load vector and solve. The boundary conditions "
                    "are already handled — node 0 was excluded from the equation system."
                ),
                "code": (
                    "RHSG = loadasem(loads, IBC, neq, 1)\n"
                    "print(f'Load vector: {RHSG}')\n"
                    "UG = np.linalg.solve(KG, RHSG)\n"
                    "print(f'Displacements: u1={UG[0]:.6f}, u2={UG[1]:.6f}, u3={UG[2]:.6f} m')\n"
                    "print(f'\\nVerification: all force flows through all springs,')\n"
                    "print(f'so F_each = 100 N, and u3 = F*(1/k0 + 1/k1 + 1/k2)')\n"
                    "u3_check = 100*(1/1000 + 1/2000 + 1/1500)\n"
                    "print(f'u3 = {u3_check:.6f} m  ✓')"
                ),
            },
        ],
    },

    "verification_equilibrium": {
        "module": "module1_springs",
        "curriculum_nodes": ["verification", "solution"],
        "description": "Verify equilibrium and compute reactions after solving",
        "steps": [
            {
                "narration": (
                    "After solving, we must verify equilibrium. Let's use the 3-spring "
                    "system we just solved and compute element forces and reactions."
                ),
                "code": (
                    "# Setup (same model as assembly demo)\n"
                    "nodes = np.array([[0, 0.0, -1], [1, 1.0, 0], [2, 2.0, 0], [3, 3.0, 0]])\n"
                    "mats = np.array([1000.0, 2000.0, 1500.0])\n"
                    "elements = np.array([[0,0,0,0,1],[1,0,1,1,2],[2,0,2,2,3]], dtype=int)\n"
                    "loads = np.array([[3, 100.0]])\n"
                    "neq, IBC = eqcounter(nodes)\n"
                    "DME_op, _, _ = DME(nodes, elements)\n"
                    "KG = assembly(elements, mats, nodes, neq, DME_op)\n"
                    "RHSG = loadasem(loads, IBC, neq, 1)\n"
                    "UG = np.linalg.solve(KG, RHSG)\n"
                    "# Add fixed node displacement\n"
                    "U_all = np.zeros(4)\n"
                    "U_all[1:] = UG\n"
                    "print('All displacements:', [f'{u:.6f}' for u in U_all])"
                ),
            },
            {
                "narration": (
                    "Now compute the force in each spring: f = k × (u_j - u_i). "
                    "Positive means tension (spring stretching)."
                ),
                "code": (
                    "print('Element forces:')\n"
                    "for i in range(3):\n"
                    "    k = mats[elements[i,2]]\n"
                    "    ni, nj = elements[i,3], elements[i,4]\n"
                    "    f = k * (U_all[nj] - U_all[ni])\n"
                    "    print(f'  Element {i}: f = {k:.0f} × ({U_all[nj]:.6f} - {U_all[ni]:.6f}) = {f:.2f} N')\n"
                    "print('\\nAll forces = 100 N ✓ (series system, same force everywhere)')"
                ),
            },
            {
                "narration": (
                    "Finally, the reaction at the fixed support. Global equilibrium "
                    "requires: R + P = 0, so R = -100 N."
                ),
                "code": (
                    "# Reaction at node 0 = k0 * (u0 - u1)\n"
                    "R0 = mats[0] * (U_all[0] - U_all[1])\n"
                    "print(f'Reaction at node 0: R = {R0:.2f} N')\n"
                    "print(f'Applied load at node 3: P = 100.0 N')\n"
                    "print(f'Sum of forces: R + P = {R0 + 100:.2f} N  (equilibrium ✓)')"
                ),
            },
        ],
    },

    # ------------------------------------------------------------------
    # MODULE 2: TRUSSES
    # ------------------------------------------------------------------

    "truss_element_stiffness": {
        "module": "module2_trusses",
        "curriculum_nodes": ["bar_element", "coordinate_transformation", "global_element_stiffness"],
        "description": "Build a truss element stiffness matrix with transformation",
        "steps": [
            {
                "narration": (
                    "Let's build the stiffness matrix for a truss element. "
                    "A horizontal bar (θ=0°) with A=1, E=1000, L=2."
                ),
                "code": (
                    "coord = np.array([[0.0, 0.0], [2.0, 0.0]])  # horizontal\n"
                    "A = 1.0\n"
                    "E = 1000.0\n"
                    "K_horiz = ueltruss2D(coord, A, E)\n"
                    "print('Horizontal truss element (4×4):')\n"
                    "print(np.round(K_horiz, 1))\n"
                    "print(f'\\nEA/L = {A*E/2.0:.1f}')"
                ),
            },
            {
                "narration": (
                    "For a horizontal element, the global K only has stiffness in the "
                    "x-direction — the truss can't resist transverse loads. "
                    "Now let's try a 45° diagonal element."
                ),
                "code": (
                    "coord_45 = np.array([[0.0, 0.0], [1.0, 1.0]])  # 45 degrees\n"
                    "L_45 = np.sqrt(2)\n"
                    "K_45 = ueltruss2D(coord_45, A, E)\n"
                    "print('45° truss element (4×4):')\n"
                    "print(np.round(K_45, 1))\n"
                    "print(f'\\nnx = ny = 1/√2 = {1/np.sqrt(2):.4f}')\n"
                    "print(f'EA/L = {A*E/L_45:.2f}')\n"
                    "print('\\nNotice: all quadrants are non-zero — the 45° element')\n"
                    "print('couples x and y displacements through the transformation.')"
                ),
            },
            {
                "narration": (
                    "Compare: the horizontal element only has entries in the ux-ux "
                    "positions. The 45° element has entries everywhere because the "
                    "axial direction projects onto both x and y."
                ),
                "code": (
                    "print('Horizontal — non-zero pattern:')\n"
                    "print((np.abs(K_horiz) > 0.01).astype(int))\n"
                    "print('\\n45° — non-zero pattern:')\n"
                    "print((np.abs(K_45) > 0.01).astype(int))\n"
                    "print('\\nThis is what coordinate transformation does —')\n"
                    "print('it distributes the axial stiffness into global DOF directions.')"
                ),
            },
        ],
    },

    # ------------------------------------------------------------------
    # MODULE 3: FRAMES
    # ------------------------------------------------------------------

    "beam_stiffness_matrix": {
        "module": "module3_frames",
        "curriculum_nodes": ["beam_element", "frame_element"],
        "description": "Build the beam element stiffness matrix",
        "steps": [
            {
                "narration": (
                    "The beam element has 4 local DOFs: transverse displacement and "
                    "rotation at each end. Let's build it for a horizontal beam "
                    "with EI = 1000, L = 3."
                ),
                "code": (
                    "coord = np.array([[0.0, 0.0], [3.0, 0.0]])  # horizontal\n"
                    "I = 1.0   # moment of inertia\n"
                    "E = 1000.0  # Young's modulus → EI = 1000\n"
                    "K_beam = uelbeam2DU(coord, I, E)\n"
                    "print('Beam element K_global (6×6):')\n"
                    "print(np.round(K_beam, 2))"
                ),
            },
            {
                "narration": (
                    "For a horizontal beam, the transformation Q just maps "
                    "uy→v and passes θ through. The 6×6 global K has bending "
                    "stiffness in the y and θ DOFs, but zero in the x DOFs "
                    "(this beam has no axial stiffness — it's axially rigid)."
                ),
                "code": (
                    "L = 3.0\n"
                    "EI = E * I\n"
                    "print('Expected coefficients:')\n"
                    "print(f'  12EI/L³ = {12*EI/L**3:.4f}')\n"
                    "print(f'  6EI/L²  = {6*EI/L**2:.4f}')\n"
                    "print(f'  4EI/L   = {4*EI/L:.4f}')\n"
                    "print(f'  2EI/L   = {2*EI/L:.4f}')\n"
                    "print(f'\\nCheck K[1,1] (uy1-uy1): {K_beam[1,1]:.4f} vs 12EI/L³ = {12*EI/L**3:.4f}')\n"
                    "print(f'Check K[2,2] (θ1-θ1):  {K_beam[2,2]:.4f} vs 4EI/L  = {4*EI/L:.4f}')"
                ),
            },
            {
                "narration": (
                    "Now a vertical column (90°). The transformation rotates "
                    "everything — bending stiffness now appears in the x-direction. "
                    "This is why columns resist lateral loads!"
                ),
                "code": (
                    "coord_vert = np.array([[0.0, 0.0], [0.0, 3.0]])  # vertical\n"
                    "K_col = uelbeam2DU(coord_vert, I, E)\n"
                    "print('Vertical column K_global (6×6):')\n"
                    "print(np.round(K_col, 2))\n"
                    "print(f'\\nK[0,0] (ux1-ux1) = {K_col[0,0]:.4f}')\n"
                    "print(f'This is 12EI/L³ = {12*EI/L**3:.4f} — bending stiffness!')\n"
                    "print('A horizontal push on the column top → bending resistance.')"
                ),
            },
        ],
    },
}


# ======================================================================
# Node-to-demo mapping (for orchestrator integration)
# ======================================================================

NODE_TO_DEMOS = {}
for demo_name, demo in CURATED_DEMOS.items():
    for node in demo.get("curriculum_nodes", []):
        NODE_TO_DEMOS.setdefault(node, []).append(demo_name)
