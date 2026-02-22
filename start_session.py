#!/usr/bin/env python3
"""Start or resume a tutoring session."""
import sys
from core.persistence_manager import PersistenceManager


def main():
    if len(sys.argv) < 2:
        # List available students
        students = PersistenceManager.list_students()
        if students:
            print("Existing students:")
            for s in students:
                print(f"  {s['student_id']}: node={s.get('current_node')}, "
                      f"sessions={s.get('session_count', 0)}")
        else:
            print("No existing students.")
        print("\nUsage: python start_session.py <student_id>")
        return

    student_id = sys.argv[1]
    pm = PersistenceManager(student_id=student_id)
    orch = pm.get_orchestrator()

    if pm.orchestrator.student.get("session_count", 0) > 1:
        summary = pm.get_resume_summary()
        print(f"\nWelcome back, {student_id}!")
        print(f"Session #{summary['session_count']}")
        print(f"You're on: {summary['current_node_title']}")
        print(f"State: {summary['current_state']}")
        print(f"Progress: {summary['progress']['mastered']}/{summary['progress']['total']} nodes mastered")
    else:
        print(f"\nWelcome, {student_id}! Starting new session.")
        print(f"First topic: {orch.node_data['title']}")

    print(f"\nStatus: {orch.get_status_summary()}")


if __name__ == "__main__":
    main()
