from pathlib import Path
import unittest

from lusas_ai.config import Settings


class ConfigTests(unittest.TestCase):
    def test_config_enables_local_automatic_upgrade(self) -> None:
        root = Path(__file__).resolve().parents[1]
        settings = Settings.load(root)

        self.assertTrue(settings.auto_model_upgrades)
        self.assertTrue(settings.evolution_enabled)
        self.assertTrue(settings.auto_code_upgrades)
        self.assertTrue(settings.auto_apply_upgrades)
        self.assertGreaterEqual(settings.model_upgrade_interval_minutes, 1)


if __name__ == "__main__":
    unittest.main()
