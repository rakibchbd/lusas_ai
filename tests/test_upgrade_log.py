from pathlib import Path
import tempfile
import unittest

from lusas_ai.upgrade_log import next_version, record


class UpgradeLogTests(unittest.TestCase):
    def test_version_increases_by_one_micro_increment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state.json"
            self.assertEqual(next_version(state), "0.000001")
            self.assertEqual(next_version(state), "0.000002")

    def test_record_writes_upgrade_event(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "upgrade.jsonl"
            record(log, status="rejected", score=0.5)
            self.assertIn('"status": "rejected"', log.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
