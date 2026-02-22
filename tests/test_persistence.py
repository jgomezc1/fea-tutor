"""Tests for student state persistence across sessions.

Verifies: auto-save, session resume, atomic writes, corruption recovery,
multi-student isolation, and resume summary generation.

Run:
    python -m pytest tests/test_persistence.py -v
"""
import pytest
import json
import shutil
import tempfile
from pathlib import Path


@pytest.fixture
def tmp_dirs():
    """Create temp directories for data and students."""
    base = Path(tempfile.mkdtemp())
    students_dir = base / "students"
    students_dir.mkdir()
    yield {"base": base, "students_dir": str(students_dir)}
    shutil.rmtree(base)


@pytest.fixture
def pm(tmp_dirs):
    """Fresh PersistenceManager for a test student."""
    from core.persistence_manager import PersistenceManager
    return PersistenceManager(
        student_id="test_student",
        data_dir="data",
        students_dir=tmp_dirs["students_dir"],
    )


@pytest.fixture
def pm_factory(tmp_dirs):
    """Factory to create PersistenceManagers sharing the same students_dir."""
    from core.persistence_manager import PersistenceManager

    def _make(student_id="test_student"):
        return PersistenceManager(
            student_id=student_id,
            data_dir="data",
            students_dir=tmp_dirs["students_dir"],
        )
    return _make


# ===================================================================
# New student creation
# ===================================================================

class TestNewStudent:
    """Test creating a fresh student."""

    def test_new_student_creates_file(self, pm):
        orch = pm.get_orchestrator()
        assert pm.exists()
        assert pm._state_file.exists()

    def test_new_student_starts_at_first_node(self, pm):
        orch = pm.get_orchestrator()
        assert orch.current_node == "element_stiffness"
        assert orch.state == "ASSESS_PRIOR"

    def test_new_student_has_metadata(self, pm):
        orch = pm.get_orchestrator()
        assert "student_id" in orch.student
        assert orch.student["student_id"] == "test_student"
        assert "created_at" in orch.student
        assert "last_session" in orch.student
        assert orch.student["session_count"] == 1

    def test_new_student_resume_summary(self, pm):
        pm.get_orchestrator()
        summary = pm.get_resume_summary()
        assert summary["student_id"] == "test_student"
        assert summary["session_count"] == 1
        assert summary["progress"]["mastered"] == 0
        assert summary["progress"]["total"] == 18


# ===================================================================
# Auto-save on state transitions
# ===================================================================

class TestAutoSave:
    """Test that state is saved after every transition."""

    def test_process_and_save_persists(self, pm):
        orch = pm.get_orchestrator()
        initial_node = orch.current_node

        # Process an evaluation
        pm.process_and_save({
            "procedural": "correct",
            "conceptual": "mechanical",
            "misconception_id": None,
            "confidence": "medium",
            "physical_reasoning_detected": False,
        })

        # Read the file back
        with open(pm._state_file) as f:
            saved = json.load(f)

        # State should have transitioned
        assert saved["current_state"] != "ASSESS_PRIOR" or saved["current_node"] != initial_node
        assert "last_saved" in saved

    def test_advance_and_save_persists(self, pm):
        orch = pm.get_orchestrator()
        # Force to ADVANCE state
        orch.state = "ADVANCE"
        result = pm.advance_and_save()
        assert result is True

        # Read back
        with open(pm._state_file) as f:
            saved = json.load(f)
        assert saved["current_node"] == "local_global_dofs"

    def test_multiple_transitions_all_saved(self, pm):
        orch = pm.get_orchestrator()

        # ASSESS_PRIOR → MODEL
        pm.process_and_save({
            "procedural": "incomplete",
            "conceptual": "absent",
            "misconception_id": None,
            "confidence": "low",
            "physical_reasoning_detected": False,
        })

        with open(pm._state_file) as f:
            saved1 = json.load(f)

        # MODEL → GP (manual transition since MODEL doesn't evaluate)
        orch.state = "GUIDED_PRACTICE"
        pm.save()

        with open(pm._state_file) as f:
            saved2 = json.load(f)

        assert saved1["last_saved"] != saved2["last_saved"]
        assert saved2["current_state"] == "GUIDED_PRACTICE"


# ===================================================================
# Session resume
# ===================================================================

class TestSessionResume:
    """Test resuming from a saved state."""

    def test_resume_preserves_node(self, pm_factory):
        # Session 1: advance to node 2
        pm1 = pm_factory("resume_test")
        orch1 = pm1.get_orchestrator()
        orch1.state = "ADVANCE"
        pm1.advance_and_save()
        assert orch1.current_node == "local_global_dofs"

        # Session 2: create new PM with same student_id
        pm2 = pm_factory("resume_test")
        orch2 = pm2.get_orchestrator()
        assert orch2.current_node == "local_global_dofs"

    def test_resume_preserves_state(self, pm_factory):
        pm1 = pm_factory("state_test")
        orch1 = pm1.get_orchestrator()
        pm1.process_and_save({
            "procedural": "correct",
            "conceptual": "mechanical",
            "misconception_id": None,
            "confidence": "medium",
            "physical_reasoning_detected": False,
        })
        saved_state = orch1.state

        pm2 = pm_factory("state_test")
        orch2 = pm2.get_orchestrator()
        assert orch2.state == saved_state

    def test_resume_preserves_mastery(self, pm_factory):
        pm1 = pm_factory("mastery_test")
        orch1 = pm1.get_orchestrator()

        # Give some correct answers to raise mastery
        for _ in range(3):
            orch1.state = "GUIDED_PRACTICE"
            pm1.process_and_save({
                "procedural": "correct",
                "conceptual": "deep",
                "misconception_id": None,
                "confidence": "high",
                "physical_reasoning_detected": True,
            })

        mastery_after = orch1.student_node["p_mastery"]

        # Resume
        pm2 = pm_factory("mastery_test")
        orch2 = pm2.get_orchestrator()
        assert abs(orch2.student_node["p_mastery"] - mastery_after) < 1e-10

    def test_resume_preserves_misconceptions(self, pm_factory):
        pm1 = pm_factory("misc_test")
        orch1 = pm1.get_orchestrator()
        orch1.state = "GUIDED_PRACTICE"
        pm1.process_and_save({
            "procedural": "correct",
            "conceptual": "mechanical",
            "misconception_id": "M2",
            "confidence": "medium",
            "physical_reasoning_detected": False,
        })

        pm2 = pm_factory("misc_test")
        orch2 = pm2.get_orchestrator()
        assert "M2" in orch2.student_node["misconceptions_observed"]

    def test_session_count_increments(self, pm_factory):
        pm1 = pm_factory("count_test")
        pm1.get_orchestrator()
        assert pm1.orchestrator.student["session_count"] == 1

        pm2 = pm_factory("count_test")
        pm2.get_orchestrator()
        assert pm2.orchestrator.student["session_count"] == 2

        pm3 = pm_factory("count_test")
        pm3.get_orchestrator()
        assert pm3.orchestrator.student["session_count"] == 3

    def test_resume_summary_for_returning_student(self, pm_factory):
        # Session 1
        pm1 = pm_factory("summary_test")
        orch1 = pm1.get_orchestrator()
        orch1.state = "ADVANCE"
        pm1.advance_and_save()

        # Session 2
        pm2 = pm_factory("summary_test")
        pm2.get_orchestrator()
        summary = pm2.get_resume_summary()
        assert summary["session_count"] == 2
        assert summary["current_node"] == "local_global_dofs"

    def test_welcome_context_for_returning_student(self, pm_factory):
        pm1 = pm_factory("welcome_test")
        orch = pm1.get_orchestrator()
        orch.state = "GUIDED_PRACTICE"
        pm1.save()

        pm2 = pm_factory("welcome_test")
        pm2.get_orchestrator()
        ctx = pm2.get_welcome_context()
        assert ctx["is_returning_student"] is True
        assert ctx["recap_needed"] is True
        assert "recap_hint" in ctx
        assert "teacher_context" in ctx


# ===================================================================
# Atomic writes and corruption recovery
# ===================================================================

class TestCorruptionRecovery:
    """Test backup and recovery mechanisms."""

    def test_backup_created_on_save(self, pm):
        orch = pm.get_orchestrator()
        pm.save()  # first save — no backup yet

        pm.save()  # second save — should create backup
        assert pm._backup_file.exists()

    def test_restore_from_backup(self, pm_factory):
        pm = pm_factory("backup_test")
        orch = pm.get_orchestrator()
        pm.save()

        # Advance and save (creates backup of original)
        orch.state = "ADVANCE"
        pm.advance_and_save()
        assert orch.current_node == "local_global_dofs"

        # Corrupt the main file
        with open(pm._state_file, "w") as f:
            f.write("corrupted!")

        # Restore from backup
        assert pm.restore_backup() is True

        # Reload — should be at previous state
        pm2 = pm_factory("backup_test")
        orch2 = pm2.get_orchestrator()
        # Backup was from before advance, so should be at element_stiffness
        assert orch2.current_node == "element_stiffness"

    def test_no_backup_returns_false(self, pm):
        assert pm.restore_backup() is False


# ===================================================================
# Multi-student isolation
# ===================================================================

class TestMultiStudent:
    """Test that multiple students don't interfere."""

    def test_separate_state_files(self, pm_factory):
        pm_a = pm_factory("alice")
        pm_b = pm_factory("bob")

        orch_a = pm_a.get_orchestrator()
        orch_b = pm_b.get_orchestrator()

        # Advance Alice
        orch_a.state = "ADVANCE"
        pm_a.advance_and_save()

        # Bob should still be at start
        assert orch_b.current_node == "element_stiffness"
        assert orch_a.current_node == "local_global_dofs"

    def test_list_students(self, pm_factory):
        pm_factory("alice").get_orchestrator()
        pm_factory("bob").get_orchestrator()
        pm_factory("carlos").get_orchestrator()

        from core.persistence_manager import PersistenceManager
        students = PersistenceManager.list_students(
            pm_factory("alice").students_dir
        )
        ids = [s["student_id"] for s in students]
        assert "alice" in ids
        assert "bob" in ids
        assert "carlos" in ids

    def test_delete_student(self, pm_factory):
        pm = pm_factory("to_delete")
        pm.get_orchestrator()
        assert pm.exists()

        pm.delete()
        assert not pm.exists()


# ===================================================================
# Export / transcript
# ===================================================================

class TestExport:
    """Test transcript export for instructor review."""

    def test_export_has_all_nodes(self, pm):
        pm.get_orchestrator()
        transcript = pm.export_transcript()
        assert transcript["student_id"] == "test_student"
        assert len(transcript["nodes"]) == 18

    def test_export_captures_progress(self, pm):
        orch = pm.get_orchestrator()
        orch.state = "GUIDED_PRACTICE"
        pm.process_and_save({
            "procedural": "correct",
            "conceptual": "deep",
            "misconception_id": None,
            "confidence": "high",
            "physical_reasoning_detected": True,
        })

        transcript = pm.export_transcript()
        node = transcript["nodes"]["element_stiffness"]
        assert node["attempts"] > 0
        assert node["p_mastery"] > 0
