"""Student state persistence for multi-session learning.

Ensures student progress survives across sessions — days, weeks, or months
apart. Three mechanisms:

  1. Auto-save: Student model written to disk after every state transition.
  2. Session resume: On startup, loads last saved state and rebuilds context.
  3. Corruption guard: Atomic writes prevent half-written files.

Usage:
    # Starting a new student
    pm = PersistenceManager(student_id="juancho_2026")
    orch = pm.get_orchestrator()

    # Resuming an existing student
    pm = PersistenceManager(student_id="juancho_2026")
    orch = pm.get_orchestrator()  # automatically loads saved progress
    print(pm.get_resume_summary())

    # After each turn (call this instead of bare process_evaluation)
    pm.process_and_save(evaluation_dict)

    # Manual checkpoint
    pm.save()
"""
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path


class PersistenceManager:
    """Manages student state persistence across sessions."""

    def __init__(
        self,
        student_id: str,
        data_dir: str = "data",
        students_dir: str = "students",
    ):
        """Initialize persistence for a student.

        Parameters
        ----------
        student_id : str
            Unique identifier for the student (e.g., "maria_2026").
            Used as the filename stem.
        data_dir : str
            Path to the tutor's data directory (curriculum, problems, etc.).
        students_dir : str
            Directory where student state files are stored.
            Created automatically if it doesn't exist.
        """
        self.student_id = student_id
        self.data_dir = data_dir
        self.students_dir = Path(students_dir)
        self.students_dir.mkdir(parents=True, exist_ok=True)

        self._state_file = self.students_dir / f"{student_id}.json"
        self._backup_file = self.students_dir / f"{student_id}.backup.json"
        self._orchestrator = None

    def get_orchestrator(self):
        """Get or create the orchestrator with loaded student state.

        Returns
        -------
        Orchestrator instance with student progress loaded.
        """
        from core.orchestrator import Orchestrator

        if self._state_file.exists():
            # Resume existing session
            self._orchestrator = Orchestrator(
                data_dir=self.data_dir,
                student_file=str(self._state_file),
                on_transition=self.save,
            )
        else:
            # New student — create from template
            self._orchestrator = Orchestrator(
                data_dir=self.data_dir,
                student_file=str(self._state_file),
                on_transition=self.save,
            )
            # Add session metadata (use assignment, not setdefault,
            # because the template may already have default values)
            self._orchestrator.student["student_id"] = self.student_id
            self._orchestrator.student["created_at"] = _now_iso()

        # Update session metadata (runs for both new and resumed)
        self._orchestrator.student["student_id"] = self.student_id
        self._orchestrator.student["last_session"] = _now_iso()
        self._orchestrator.student.setdefault("session_count", 0)
        self._orchestrator.student["session_count"] += 1
        self.save()

        return self._orchestrator

    @property
    def orchestrator(self):
        """The current orchestrator. Call get_orchestrator() first."""
        if self._orchestrator is None:
            return self.get_orchestrator()
        return self._orchestrator

    # ------------------------------------------------------------------
    # Core persistence operations
    # ------------------------------------------------------------------

    def save(self):
        """Persist student state to disk with atomic write.

        Uses write-to-temp-then-rename to prevent corruption if the
        process crashes mid-write.
        """
        if self._orchestrator is None:
            return

        # Add timestamp
        self._orchestrator.student["last_saved"] = _now_iso()

        data = self._orchestrator.student
        tmp_file = self._state_file.with_suffix(".tmp")

        try:
            # Write to temp file
            with open(tmp_file, "w") as f:
                json.dump(data, f, indent=2)

            # Backup current state before overwriting
            if self._state_file.exists():
                shutil.copy2(self._state_file, self._backup_file)

            # Atomic rename (on POSIX; on Windows this is close enough)
            os.replace(tmp_file, self._state_file)

        except Exception as e:
            # Clean up temp file on failure
            if tmp_file.exists():
                tmp_file.unlink()
            raise IOError(f"Failed to save student state: {e}") from e

    def process_and_save(self, evaluation: dict):
        """Process an evaluation and auto-save.

        This is the primary method for advancing the tutor. It wraps
        orchestrator.process_evaluation() with automatic persistence.

        Parameters
        ----------
        evaluation : dict
            Evaluator output with keys: procedural, conceptual,
            misconception_id, confidence, physical_reasoning_detected
        """
        orch = self.orchestrator
        orch.process_evaluation(evaluation)
        self.save()

    def advance_and_save(self) -> bool:
        """Advance to next node and auto-save.

        Returns
        -------
        bool : True if advanced, False if curriculum complete.
        """
        orch = self.orchestrator
        result = orch.advance_to_next_node()
        self.save()
        return result

    # ------------------------------------------------------------------
    # Session resume helpers
    # ------------------------------------------------------------------

    def get_resume_summary(self) -> dict:
        """Get a summary of where the student left off.

        Useful for the teacher to generate a "Welcome back" message
        at the start of a new session.

        Returns
        -------
        dict with: student_id, current_node, current_state, last_session,
                   session_count, progress (nodes mastered / total),
                   current_node_summary
        """
        orch = self.orchestrator
        student = orch.student

        # Count mastered nodes
        mastered = []
        in_progress = []
        not_started = []
        for node_id, node_data in student.get("nodes", {}).items():
            if node_data.get("level0_passed"):
                mastered.append(node_id)
            elif node_data.get("attempts", 0) > 0:
                in_progress.append(node_id)
            else:
                not_started.append(node_id)

        # Current node details
        sn = orch.student_node
        unresolved = [
            m for m in sn["misconceptions_observed"]
            if m not in sn["misconceptions_resolved"]
        ]

        return {
            "student_id": self.student_id,
            "current_node": orch.current_node,
            "current_node_title": orch.node_data["title"],
            "current_state": orch.state,
            "scaffolding_level": orch.scaffolding_level,
            "p_mastery": sn["p_mastery"],
            "attempts_on_current": sn["attempts"],
            "unresolved_misconceptions": unresolved,
            "last_session": student.get("last_session"),
            "session_count": student.get("session_count", 0),
            "progress": {
                "mastered": len(mastered),
                "in_progress": len(in_progress),
                "not_started": len(not_started),
                "total": len(student.get("nodes", {})),
                "mastered_nodes": mastered,
            },
        }

    def get_welcome_context(self) -> dict:
        """Build context for the teacher to generate a welcome-back message.

        Includes resume summary plus the appropriate CAM technique
        and problem for the current state.
        """
        summary = self.get_resume_summary()
        orch = self.orchestrator

        context = {
            "is_returning_student": summary["session_count"] > 1,
            "resume_summary": summary,
            "teacher_context": orch.build_teacher_context(),
        }

        # If the student was mid-problem, include a recap hint
        if summary["current_state"] in ("GUIDED_PRACTICE", "DIAGNOSE"):
            context["recap_needed"] = True
            context["recap_hint"] = (
                f"The student was working on {summary['current_node_title']} "
                f"in {summary['current_state']} state with scaffolding level "
                f"{summary['scaffolding_level']}. They had {summary['attempts_on_current']} "
                f"attempts. Resume where they left off."
            )
        elif summary["current_state"] == "MODEL":
            context["recap_needed"] = False
            context["recap_hint"] = (
                f"The student is about to see the demonstration for "
                f"{summary['current_node_title']}. This may be the start of "
                f"a new concept."
            )
        else:
            context["recap_needed"] = False

        return context

    # ------------------------------------------------------------------
    # Admin / debugging
    # ------------------------------------------------------------------

    def exists(self) -> bool:
        """Whether a saved state exists for this student."""
        return self._state_file.exists()

    def delete(self):
        """Delete all saved state for this student. Irreversible."""
        for f in [self._state_file, self._backup_file,
                  self._state_file.with_suffix(".tmp")]:
            if f.exists():
                f.unlink()
        self._orchestrator = None

    def restore_backup(self) -> bool:
        """Restore from backup if main state file is corrupted.

        Returns True if backup was restored, False if no backup exists.
        """
        if self._backup_file.exists():
            shutil.copy2(self._backup_file, self._state_file)
            self._orchestrator = None  # force reload
            return True
        return False

    def export_transcript(self) -> dict:
        """Export a summary of the student's full journey.

        Useful for instructor review or research data collection.
        """
        orch = self.orchestrator
        student = orch.student
        transcript = {
            "student_id": self.student_id,
            "exported_at": _now_iso(),
            "session_count": student.get("session_count", 0),
            "created_at": student.get("created_at"),
            "last_session": student.get("last_session"),
            "current_node": orch.current_node,
            "current_state": orch.state,
            "nodes": {},
        }

        for node_id, node_data in student.get("nodes", {}).items():
            transcript["nodes"][node_id] = {
                "p_mastery": node_data.get("p_mastery"),
                "attempts": node_data.get("attempts", 0),
                "level0_passed": node_data.get("level0_passed", False),
                "misconceptions_observed": node_data.get("misconceptions_observed", []),
                "misconceptions_resolved": node_data.get("misconceptions_resolved", []),
                "lo_status": node_data.get("lo_status", {}),
                "scaffolding_history": node_data.get("scaffolding_history", []),
            }

        return transcript

    @classmethod
    def list_students(cls, students_dir: str = "students") -> list[dict]:
        """List all saved students with basic info.

        Returns list of dicts with: student_id, last_session, current_node
        """
        sdir = Path(students_dir)
        if not sdir.exists():
            return []

        students = []
        for f in sorted(sdir.glob("*.json")):
            if f.stem.endswith(".backup") or f.stem.endswith(".tmp"):
                continue
            try:
                with open(f) as fh:
                    data = json.load(fh)
                students.append({
                    "student_id": data.get("student_id", f.stem),
                    "last_session": data.get("last_session"),
                    "current_node": data.get("current_node"),
                    "current_state": data.get("current_state"),
                    "session_count": data.get("session_count", 0),
                })
            except (json.JSONDecodeError, KeyError):
                students.append({
                    "student_id": f.stem,
                    "error": "corrupted state file",
                })

        return students


# ======================================================================
# Helpers
# ======================================================================

def _now_iso() -> str:
    """Current UTC time as ISO string."""
    return datetime.now(timezone.utc).isoformat()
