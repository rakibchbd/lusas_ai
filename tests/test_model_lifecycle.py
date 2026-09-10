from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from training.model_lifecycle import promote, restore_backup


class ModelLifecycleTests(unittest.TestCase):
    def test_promotion_backs_up_previous_model_and_can_restore(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate_one = root / "models" / "candidates" / "one"
            candidate_two = root / "models" / "candidates" / "two"
            candidate_one.mkdir(parents=True)
            candidate_two.mkdir(parents=True)
            (candidate_one / "weights.bin").write_text("one", encoding="utf-8")
            (candidate_two / "weights.bin").write_text("two", encoding="utf-8")

            first_backup = promote(candidate_one, root, evaluation_passed=True)
            self.assertFalse((first_backup / "weights.bin").exists())
            second_backup = promote(candidate_two, root, evaluation_passed=True)
            self.assertEqual(
                (second_backup / "weights.bin").read_text(encoding="utf-8"),
                "one",
            )

            restore_backup(second_backup, root)
            self.assertEqual(
                (root / "models" / "production" / "weights.bin").read_text(
                    encoding="utf-8"
                ),
                "one",
            )


if __name__ == "__main__":
    unittest.main()
