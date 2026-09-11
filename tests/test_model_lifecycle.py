from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from training.model_lifecycle import promote, restore_backup


EVALUATION = {
    "passed": True,
    "gates": {
        "accuracy": True,
        "hallucination": True,
        "safety": True,
        "regression": True,
        "english": True,
        "bengali": True,
        "latency": True,
        "resources": True,
    },
}


class ModelLifecycleTests(unittest.TestCase):
    def test_promotion_backs_up_previous_model_and_can_restore(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate_one = root / "models" / "sara-1.0" / "candidates" / "one"
            candidate_two = root / "models" / "sara-1.0" / "candidates" / "two"
            candidate_one.mkdir(parents=True)
            candidate_two.mkdir(parents=True)
            (candidate_one / "weights.bin").write_text("one", encoding="utf-8")
            (candidate_two / "weights.bin").write_text("two", encoding="utf-8")

            first_backup = promote(candidate_one, root, evaluation_passed=True, model_id="sara-1.0", admin_approved=True, evaluation=EVALUATION, approved_by="test-admin")
            self.assertFalse((first_backup / "weights.bin").exists())
            second_backup = promote(candidate_two, root, evaluation_passed=True, model_id="sara-1.0", admin_approved=True, evaluation=EVALUATION, approved_by="test-admin")
            self.assertEqual(
                (second_backup / "weights.bin").read_text(encoding="utf-8"),
                "one",
            )

            restore_backup(second_backup, root, model_id="sara-1.0")
            self.assertEqual(
                (root / "models" / "sara-1.0" / "stable" / "weights.bin").read_text(
                    encoding="utf-8"
                ),
                "one",
            )


if __name__ == "__main__":
    unittest.main()
