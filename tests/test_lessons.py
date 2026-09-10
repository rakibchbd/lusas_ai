from pathlib import Path
import tempfile
import unittest

from lusas_ai.lessons import record_failure


class LessonTests(unittest.TestCase):
    def test_failure_creates_lesson_and_regression_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lesson = record_failure(Path(temporary), "training", "out of memory")
            self.assertTrue(lesson["fingerprint"])
            self.assertIn("out of memory", (Path(temporary) / ".lusas" / "lessons.jsonl").read_text())
            self.assertIn("training", (Path(temporary) / ".lusas" / "regressions.jsonl").read_text())


if __name__ == "__main__":
    unittest.main()
