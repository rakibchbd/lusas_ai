from pathlib import Path
import json
import tempfile
import unittest

from lusas_ai.config import Settings


class ConfigTests(unittest.TestCase):
    def test_config_enables_local_automatic_upgrade(self) -> None:
        root = Path(__file__).resolve().parents[1]
        settings = Settings.load(root)

        self.assertTrue(settings.auto_model_upgrades)
        self.assertTrue(settings.evolution_enabled)
        self.assertTrue(settings.auto_code_upgrades)
        self.assertTrue(settings.continuous_learning_enabled)
        self.assertEqual(settings.background_max_concurrent_jobs, 1)
        self.assertFalse(settings.auto_apply_upgrades)
        self.assertGreaterEqual(settings.model_upgrade_interval_minutes, 1)
        self.assertEqual(settings.default_model_id, "sara-1.0")
        self.assertEqual(settings.model_storage, "models")

    def test_config_paths_cannot_escape_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "learning_file": "../outside.json",
                        "workspace": "/tmp/outside",
                        "model_upgrade_interval_minutes": -4,
                        "web_max_items": 0,
                    }
                ),
                encoding="utf-8",
            )
            settings = Settings.load(root)
            self.assertEqual(settings.learning_file, ".lusas/learned.jsonl")
            self.assertEqual(settings.workspace, "workspace")
            self.assertEqual(settings.model_upgrade_interval_minutes, 1)
            self.assertEqual(settings.web_max_items, 1)


if __name__ == "__main__":
    unittest.main()
