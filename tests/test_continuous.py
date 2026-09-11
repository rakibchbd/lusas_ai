from pathlib import Path
import json
import tempfile
import unittest

from lusas_ai.admin_db import AdminStore
from lusas_ai.config import Settings
from lusas_ai.continuous import incremental_ingestion, maintain


class ContinuousLearningTests(unittest.TestCase):
    def test_incremental_ingestion_only_reports_new_file_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = Settings(root=root)
            settings.learning_path.parent.mkdir(parents=True, exist_ok=True)
            settings.learning_path.write_text('{"instruction":"one","output":"two"}\n', encoding="utf-8")
            first = incremental_ingestion(settings)
            second = incremental_ingestion(settings)
            self.assertGreater(first["changed_count"], 0)
            self.assertEqual(second["changed_count"], 0)
            settings.learning_path.write_text(
                '{"instruction":"one","output":"two"}\n{"instruction":"three","output":"four"}\n',
                encoding="utf-8",
            )
            third = incremental_ingestion(settings)
            self.assertIn(str(settings.learning_path), third["changed_files"])

    def test_maintenance_runs_real_retention_checks_and_persists_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            settings = Settings(root=Path(temporary))
            result = maintain(settings, evolution_id="EV-test")
            self.assertEqual(result["status"], "completed")
            self.assertGreaterEqual(result["practice"]["attempted"], 1)
            self.assertTrue(settings.continuous_state_path.exists())
            self.assertTrue(settings.practice_path.exists())
            state = json.loads(settings.continuous_state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["last_evolution_id"], "EV-test")

    def test_failed_interaction_becomes_a_gap_signal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            settings = Settings(root=Path(temporary))
            store = AdminStore(settings)
            interaction = store.record_interaction(
                "sara-1.0", "What is an unknown thing?", "I don't know that yet."
            )
            result = maintain(settings)
            self.assertEqual(result["interactions"]["processed"], 1)
            self.assertEqual(result["interactions"]["failures"], 1)
            self.assertTrue(settings.knowledge_gaps_path.exists())
            self.assertTrue(interaction["interaction_id"])
            self.assertEqual(store.continuous_counts()["queued_interactions"], 0)


if __name__ == "__main__":
    unittest.main()
