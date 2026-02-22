"""Notebook knowledge retriever for the FEA tutor.

Provides the teacher agent with code references, expert narrations,
and cross-module comparisons from the course notebooks.
"""

import json
from pathlib import Path


class NotebookKnowledge:
    """Retrieves notebook code and narrations for the teacher agent."""

    def __init__(self, data_dir="data"):
        map_path = Path(data_dir) / "notebook_function_map.json"
        kb_path = Path(data_dir) / "notebook_reference_kb.json"
        with open(map_path) as f:
            self.function_map = json.load(f)
        with open(kb_path) as f:
            self.kb = json.load(f)
        # Build reverse index: curriculum_node → list of kb entry keys
        self._node_index = {}
        for key, entry in self.kb["notebook_references"].items():
            for node in entry["curriculum_nodes"]:
                self._node_index.setdefault(node, []).append(key)

    def get_references_for_node(self, node_id: str) -> list[dict]:
        """Return all notebook references relevant to a curriculum node.

        Returns list of dicts with keys: code, expert_narration, module,
        function_family, what_student_should_notice, questions_the_tutor_can_ask
        """
        keys = self._node_index.get(node_id, [])
        return [self.kb["notebook_references"][k] for k in keys]

    def get_cross_module_narrative(self, function_family: str) -> str | None:
        """Return the cross-module evolution narrative for a function family.

        Use during REFLECTION state to show how the same function evolves
        across springs → trusses → frames.
        """
        family = self.function_map["function_families"].get(function_family)
        if family:
            return family.get("cross_module_narrative")
        return None

    def get_evolution_table(self) -> dict:
        """Return the cross-module evolution summary table.

        Use during REFLECTION or ADVANCE states to show the big picture.
        """
        return self.kb.get("cross_module_evolution_table", {})

    def get_code_for_module_and_family(self, module: str, function_family: str) -> dict | None:
        """Return the specific code variant for a module and function family.

        E.g., get_code_for_module_and_family("module2_trusses", "uelXXX")
        returns the ueltruss2D entry.
        """
        module_to_nb = {
            "module1_springs": "nb01",
            "module2_trusses": "nb02",
            "module3_frames": "nb03",
        }
        family = self.function_map["function_families"].get(function_family)
        if family:
            nb_key = module_to_nb.get(module, module)
            return family["variants"].get(nb_key)
        return None

    def get_curriculum_difference_note(self, node_id: str) -> str | None:
        """Check if there's a notebook-vs-curriculum difference for a node.

        Returns the teaching note if applicable (e.g., the frame element
        notebook uses beam-only while curriculum uses full frame).
        """
        refs = self.get_references_for_node(node_id)
        for ref in refs:
            diff = ref.get("curriculum_vs_notebook_difference")
            if diff:
                return diff.get("teaching_note")
        return None
