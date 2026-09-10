from pathlib import Path
import tempfile
import unittest

from lusas_ai.learning import LearningStore


class LearningStoreTests(unittest.TestCase):
    def test_stores_and_reads_approved_examples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = LearningStore(Path(temporary) / ".lusas" / "learned.jsonl")
            store.add("Who made you?", "I am Lusa, created by Rakib Chowdhury.")
            self.assertEqual(
                store.examples(),
                [
                    {
                        "instruction": "Who made you?",
                        "output": "I am Lusa, created by Rakib Chowdhury.",
                    }
                ],
            )

    def test_rejects_empty_examples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = LearningStore(Path(temporary) / "learned.jsonl")
            with self.assertRaises(ValueError):
                store.add("", "answer")


if __name__ == "__main__":
    unittest.main()
