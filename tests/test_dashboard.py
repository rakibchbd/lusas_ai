from pathlib import Path
import tempfile
import unittest

from lusas_ai.config import Settings
from lusas_ai.dashboard import render, write


class DashboardTests(unittest.TestCase):
    def test_dashboard_is_local_and_escapes_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            settings = Settings(root=Path(temporary))
            content = render(settings)
            self.assertIn("LUSAS AI", content)
            self.assertIn("v0.000", content)
            self.assertNotIn("<script>alert", content)
            self.assertTrue(write(settings).exists())


if __name__ == "__main__":
    unittest.main()
