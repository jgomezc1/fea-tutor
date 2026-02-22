"""Tests for the notebook knowledge retriever."""

import json
import unittest
from pathlib import Path
from core.notebook_knowledge import NotebookKnowledge


class TestNotebookKnowledgeLoading(unittest.TestCase):
    """Verify both JSON files load without errors."""

    def test_loads_without_error(self):
        nb = NotebookKnowledge()
        self.assertIsNotNone(nb.function_map)
        self.assertIsNotNone(nb.kb)

    def test_function_map_has_families(self):
        nb = NotebookKnowledge()
        families = nb.function_map.get("function_families", {})
        self.assertGreater(len(families), 0)

    def test_kb_has_references(self):
        nb = NotebookKnowledge()
        refs = nb.kb.get("notebook_references", {})
        self.assertGreater(len(refs), 0)


class TestNodeIndex(unittest.TestCase):
    """For each curriculum node that has a notebook reference,
    verify get_references_for_node returns at least one entry with non-empty code."""

    def setUp(self):
        self.nb = NotebookKnowledge()
        # Collect all curriculum nodes referenced in the KB
        self.referenced_nodes = set()
        for entry in self.nb.kb["notebook_references"].values():
            for node in entry["curriculum_nodes"]:
                self.referenced_nodes.add(node)

    def test_each_referenced_node_has_entries(self):
        for node in self.referenced_nodes:
            refs = self.nb.get_references_for_node(node)
            self.assertGreater(
                len(refs), 0,
                f"Node '{node}' is referenced in KB but get_references_for_node returns nothing"
            )

    def test_each_referenced_node_has_code(self):
        for node in self.referenced_nodes:
            refs = self.nb.get_references_for_node(node)
            has_code = any(ref.get("code") for ref in refs)
            self.assertTrue(
                has_code,
                f"Node '{node}' has no reference with non-empty code"
            )


class TestCrossModuleNarrative(unittest.TestCase):
    """Verify all 6 function families have cross-module narratives."""

    EXPECTED_FAMILIES = [
        "uelXXX", "eqcounter", "DME", "assembly", "loadasem", "main_program"
    ]

    def setUp(self):
        self.nb = NotebookKnowledge()

    def test_all_families_have_narratives(self):
        for family in self.EXPECTED_FAMILIES:
            narrative = self.nb.get_cross_module_narrative(family)
            self.assertIsNotNone(
                narrative,
                f"Family '{family}' has no cross-module narrative"
            )
            self.assertGreater(len(narrative), 0)

    def test_narrative_count(self):
        families = self.nb.function_map["function_families"]
        self.assertEqual(len(families), 6)


class TestEvolutionTable(unittest.TestCase):
    """Verify the table has 6 rows and the correct column structure."""

    def setUp(self):
        self.nb = NotebookKnowledge()
        self.table = self.nb.get_evolution_table()

    def test_table_exists(self):
        self.assertIsNotNone(self.table)
        self.assertIn("rows", self.table)
        self.assertIn("columns", self.table)

    def test_table_has_six_rows(self):
        self.assertEqual(len(self.table["rows"]), 6)

    def test_columns_structure(self):
        cols = self.table["columns"]
        self.assertEqual(len(cols), 5)
        self.assertIn("function", cols)
        self.assertIn("what changes", cols)

    def test_each_row_has_five_columns(self):
        for row in self.table["rows"]:
            self.assertEqual(
                len(row), 5,
                f"Row '{row[0]}' has {len(row)} columns, expected 5"
            )


class TestCurriculumDifference(unittest.TestCase):
    """Verify that beam_element and frame_element nodes return a
    curriculum difference note."""

    def setUp(self):
        self.nb = NotebookKnowledge()

    def test_beam_element_has_difference(self):
        note = self.nb.get_curriculum_difference_note("beam_element")
        self.assertIsNotNone(note)
        self.assertGreater(len(note), 0)

    def test_frame_element_has_difference(self):
        note = self.nb.get_curriculum_difference_note("frame_element")
        self.assertIsNotNone(note)
        self.assertGreater(len(note), 0)

    def test_spring_node_has_no_difference(self):
        note = self.nb.get_curriculum_difference_note("element_stiffness")
        self.assertIsNone(note)


class TestAllNodesReachable(unittest.TestCase):
    """For every node in the curriculum graph, check whether notebook
    references exist. Not all nodes need them, but key nodes SHOULD."""

    NODES_THAT_SHOULD_HAVE_REFS = [
        "element_stiffness",
        "local_global_dofs",
        "assembly",
        "boundary_conditions",
        "bar_element",
        "coordinate_transformation",
        "global_element_stiffness",
        "truss_assembly_solution",
        "beam_element",
        "frame_element",
        "frame_transformation",
        "frame_global_stiffness",
        "frame_assembly_solution",
    ]

    def setUp(self):
        self.nb = NotebookKnowledge()
        with open(Path("data") / "curriculum_graph.json") as f:
            self.curriculum = json.load(f)

    def test_key_nodes_have_references(self):
        for node in self.NODES_THAT_SHOULD_HAVE_REFS:
            refs = self.nb.get_references_for_node(node)
            self.assertGreater(
                len(refs), 0,
                f"Key node '{node}' should have notebook references but has none"
            )

    def test_all_referenced_nodes_exist_in_curriculum(self):
        """Every node referenced in the KB should exist in the curriculum graph."""
        curriculum_nodes = set(self.curriculum["nodes"].keys())
        for entry in self.nb.kb["notebook_references"].values():
            for node in entry["curriculum_nodes"]:
                self.assertIn(
                    node, curriculum_nodes,
                    f"KB references node '{node}' which does not exist in curriculum"
                )


class TestCodeForModuleAndFamily(unittest.TestCase):
    """Test the get_code_for_module_and_family method."""

    def setUp(self):
        self.nb = NotebookKnowledge()

    def test_spring_element(self):
        result = self.nb.get_code_for_module_and_family("module1_springs", "uelXXX")
        self.assertIsNotNone(result)
        self.assertEqual(result["function_name"], "uelspring")

    def test_truss_element(self):
        result = self.nb.get_code_for_module_and_family("module2_trusses", "uelXXX")
        self.assertIsNotNone(result)
        self.assertEqual(result["function_name"], "ueltruss2D")

    def test_frame_element(self):
        result = self.nb.get_code_for_module_and_family("module3_frames", "uelXXX")
        self.assertIsNotNone(result)
        self.assertEqual(result["function_name"], "uelbeam2DU")

    def test_invalid_family_returns_none(self):
        result = self.nb.get_code_for_module_and_family("module1_springs", "nonexistent")
        self.assertIsNone(result)


class TestOrchestratorNotebookContext(unittest.TestCase):
    """Verify the orchestrator includes notebook references in teacher context."""

    def setUp(self):
        from core.orchestrator import Orchestrator
        self.orch = Orchestrator()

    def test_context_has_notebook_references(self):
        context = self.orch.build_teacher_context()
        self.assertIn("notebook_references", context)
        self.assertIsInstance(context["notebook_references"], list)

    def test_context_has_curriculum_difference(self):
        context = self.orch.build_teacher_context()
        self.assertIn("curriculum_difference", context)

    def test_element_stiffness_has_references(self):
        """element_stiffness is the default starting node and should have refs."""
        context = self.orch.build_teacher_context()
        self.assertGreater(len(context["notebook_references"]), 0)

    def test_advance_state_has_evolution_table(self):
        """ADVANCE state should include the evolution table."""
        self.orch.state = "ADVANCE"
        context = self.orch.build_teacher_context()
        self.assertIn("evolution_table", context)
        self.assertIn("rows", context["evolution_table"])


if __name__ == "__main__":
    unittest.main()
