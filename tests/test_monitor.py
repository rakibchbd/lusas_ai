from pathlib import Path
import tempfile
import unittest

from lusas_ai.config import Settings
from lusas_ai.learning import LearningStore
from lusas_ai.monitor import report, snapshot


class MonitorTests(unittest.TestCase):
    def test_snapshot_reports_learned_examples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = Settings(root=root)
            LearningStore(settings.learning_path).add("question", "answer")
            output = snapshot(settings)
            self.assertIn("Learned examples: 1", output)
            self.assertIn("question", output)

    def test_report_exposes_policy_and_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = Settings(root=root)
            settings.progress_path.parent.mkdir(parents=True, exist_ok=True)
            settings.progress_path.write_text(
                '{"phase":"tests","message":"running","status":"running"}\n',
                encoding="utf-8",
            )
            payload = report(settings)
            self.assertFalse(payload["policy"]["git_commit_upgrades"])
            self.assertEqual(payload["recent_progress"][0]["phase"], "tests")


if __name__ == "__main__":
    unittest.main()
