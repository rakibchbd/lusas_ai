from pathlib import Path
import tempfile
import unittest

from lusas_ai.upgrade_log import next_version, record
from lusas_ai.version_registry import append_version, bootstrap_legacy, read_versions


class UpgradeLogTests(unittest.TestCase):
    def test_version_increases_by_one_micro_increment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state.json"
            self.assertEqual(next_version(state), "v0.001")
            self.assertEqual(next_version(state), "v0.002")

    def test_record_writes_upgrade_event(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "upgrade.jsonl"
            record(log, status="rejected", score=0.5)
            self.assertIn('"status": "rejected"', log.read_text(encoding="utf-8"))

    def test_version_registry_round_trips_required_deployment_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "versions.jsonl"
            append_version(
                path,
                version="v0.001",
                parent_version=None,
                score_before=0.0,
                score_after=1.0,
                rollback_point="models/backups/one",
            )
            version = read_versions(path)[0]
            self.assertEqual(version["status"], "DEPLOYED")
            self.assertEqual(version["version"], "v0.001")
            self.assertEqual(version["rollback_point"], "models/backups/one")

    def test_version_registry_imports_legacy_promotions_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "versions.jsonl"
            versions = bootstrap_legacy(
                path,
                [{"status": "promoted", "version": "0.000004", "score": 1.0}],
            )
            self.assertEqual(versions[0]["version"], "v0.004")
            self.assertEqual(versions[0]["deployment_status"], "legacy-imported")


if __name__ == "__main__":
    unittest.main()
