from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from lusas_ai.config import Settings
from lusas_ai.control import update as update_control
from lusas_ai.cycle_state import fingerprint, write as write_cycle_state
from training import auto_upgrade


class AutomaticUpgradeTests(unittest.TestCase):
    def test_emergency_stop_prevents_a_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = Settings(root=root)
            update_control(root, settings.autonomy_level, emergency_stop=True)
            with patch.object(auto_upgrade.Settings, "load", return_value=settings):
                result = auto_upgrade.run_once(root)
            self.assertEqual(result["status"], "stopped")
            self.assertFalse(settings.evolution_cycles_path.exists())

    def test_code_evolution_runs_when_model_inputs_are_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            examples = root / "training" / "data" / "examples.jsonl"
            evaluation = root / "training" / "data" / "eval.jsonl"
            examples.parent.mkdir(parents=True)
            examples.write_text('{"instruction":"one","output":"two"}\n', encoding="utf-8")
            evaluation.write_text(
                '{"instruction":"one","expected":"two"}\n', encoding="utf-8"
            )
            settings = Settings(
                root=root,
                auto_model_upgrades=True,
                evolution_enabled=True,
                auto_code_upgrades=True,
                web_learning_enabled=False,
            )
            write_cycle_state(
                settings.cycle_state_path,
                {
                    "input_fingerprint": fingerprint(
                        [
                            examples,
                            settings.learning_path,
                            settings.web_training_path,
                            settings.web_cache_path,
                            evaluation,
                        ]
                    ),
                    "learned_examples": 1,
                },
            )
            fake_agent = SimpleNamespace(
                evolve=lambda *args, **kwargs: SimpleNamespace(
                    status="rejected", reason="no safe improvement"
                )
            )
            with (
                patch.object(auto_upgrade.Settings, "load", return_value=settings),
                patch.object(auto_upgrade, "LusasAgent", return_value=fake_agent),
            ):
                result = auto_upgrade.run_once(root)

            self.assertEqual(result["model_upgrade"]["status"], "skipped")
            self.assertEqual(result["code_upgrade"]["status"], "rejected")
            self.assertEqual(result["status"], "rejected")


if __name__ == "__main__":
    unittest.main()
