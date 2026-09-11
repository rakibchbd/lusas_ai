import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "training" / "data" / "examples.jsonl"


class TrainingDataTests(unittest.TestCase):
    def test_human_question_corpus_is_valid_and_broad(self) -> None:
        records = [
            json.loads(line)
            for line in DATASET.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertGreaterEqual(len(records), 45)
        self.assertTrue(all(record.get("instruction") and record.get("output") for record in records))

        prompts = {record["instruction"] for record in records}
        for prompt in (
            "Hello there",
            "What do you know?",
            "Who is Rakin Hasan?",
            "Explain Python simply.",
            "Write a Python function that checks whether a number is even.",
            "Can you hack into a website?",
            "Could you check my code?",
        ):
            with self.subTest(prompt=prompt):
                self.assertIn(prompt, prompts)


if __name__ == "__main__":
    unittest.main()
