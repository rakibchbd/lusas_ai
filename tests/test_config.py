from pathlib import Path
import unittest

from lusas_ai.config import Settings


class ConfigTests(unittest.TestCase):
    def test_default_policy_keeps_source_evolution_staged(self) -> None:
        root = Path(__file__).resolve().parents[1]
        settings = Settings.load(root)

        self.assertTrue(settings.auto_model_upgrades)
        self.assertFalse(settings.evolution_enabled)
        self.assertFalse(settings.auto_code_upgrades)
        self.assertFalse(settings.auto_apply_upgrades)
        self.assertFalse(settings.git_commit_upgrades)
        self.assertGreaterEqual(settings.model_upgrade_interval_minutes, 1)


if __name__ == "__main__":
    unittest.main()
