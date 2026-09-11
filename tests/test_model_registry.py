from pathlib import Path
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from lusas_ai.config import Settings
from lusas_ai.model_registry import catalog, foundation_config, select_model, selected_model_id


class ModelRegistryTests(unittest.TestCase):
    def test_catalog_contains_only_the_two_official_models(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            settings = Settings(root=Path(temporary))
            models = catalog(settings)
            self.assertEqual([model["model_id"] for model in models], ["sara-1.0", "lira-1.0"])
            self.assertEqual([model["display_name"] for model in models], ["Sara 1.0", "Lira 1.0"])

    def test_selection_is_persisted_and_foundation_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            settings = Settings(root=Path(temporary), foundation_models={"lira-1.0": {"model": "approved/lira", "path": ""}})
            self.assertEqual(selected_model_id(settings), "sara-1.0")
            select_model(settings, "lira-1.0")
            self.assertEqual(selected_model_id(settings), "lira-1.0")
            self.assertEqual(json.loads(settings.selected_model_path.read_text())["model_id"], "lira-1.0")
            self.assertEqual(foundation_config(settings, "sara-1.0"), {"model": None, "path": None})
            with patch.dict(os.environ, {"LUSAS_LIRA_FOUNDATION_MODEL": "env/lira"}, clear=False):
                self.assertEqual(foundation_config(settings, "lira-1.0")["model"], "env/lira")


if __name__ == "__main__":
    unittest.main()
