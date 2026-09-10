from pathlib import Path
import tempfile
import unittest

from lusas_ai.web_learning import WebCollector


class WebLearningTests(unittest.TestCase):
    def test_only_configured_official_hosts_are_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            collector = WebCollector(Path(temporary) / "pending.jsonl")
            self.assertTrue(collector._allowed("https://docs.python.org/3/"))
            self.assertFalse(collector._allowed("https://example.com/"))


if __name__ == "__main__":
    unittest.main()
