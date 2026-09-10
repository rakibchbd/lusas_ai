from pathlib import Path
import tempfile
import unittest

from lusas_ai.config import Settings
from lusas_ai.learning import LearningStore
from lusas_ai.monitor import snapshot


class MonitorTests(unittest.TestCase):
    def test_snapshot_reports_learned_examples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = Settings(root=root)
            LearningStore(settings.learning_path).add("question", "answer")
            output = snapshot(settings)
            self.assertIn("Learned examples: 1", output)
            self.assertIn("question", output)


if __name__ == "__main__":
    unittest.main()
