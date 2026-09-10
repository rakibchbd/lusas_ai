from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from lusas_ai.config import Settings
from lusas_ai.updater import create_backup, perform_upgrade, stage_candidate


class UpdaterTests(unittest.TestCase):
    def test_backup_contains_managed_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "lusas_ai" / "example.py"
            source.parent.mkdir()
            source.write_text("print('old')\n", encoding="utf-8")

            backup = create_backup(root)

            self.assertEqual(
                (backup / "lusas_ai" / "example.py").read_text(encoding="utf-8"),
                "print('old')\n",
            )
            self.assertTrue((backup / "manifest.json").exists())

    def test_stage_rejects_paths_outside_managed_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(ValueError):
                stage_candidate(root, {"config.json": "{}"})
            with self.assertRaises(ValueError):
                stage_candidate(root, {"lusas_ai/.git/config": "unsafe"})

    def test_upgrade_tests_and_applies_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "lusas_ai"
            tests = root / "tests"
            package.mkdir()
            tests.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "value.py").write_text("VALUE = 1\n", encoding="utf-8")
            (tests / "test_value.py").write_text(
                "from unittest import TestCase\n"
                "from lusas_ai.value import VALUE\n\n"
                "class ValueTests(TestCase):\n"
                "    def test_value(self):\n"
                "        self.assertEqual(VALUE, 2)\n",
                encoding="utf-8",
            )
            settings = Settings(root=root, notify_file=".lusas/test.jsonl")

            result = perform_upgrade(
                settings,
                {"lusas_ai/value.py": "VALUE = 2\n"},
                goal="update test value",
                apply=True,
            )

            self.assertTrue(result.tests.passed)
            self.assertTrue(result.applied)
            self.assertEqual(
                (package / "value.py").read_text(encoding="utf-8"),
                "VALUE = 2\n",
            )
            self.assertTrue(result.backup_path.exists())


if __name__ == "__main__":
    unittest.main()
