from pathlib import Path
import unittest

from lusas_ai.service import SERVICE_LABEL, build_plist


class ServiceTests(unittest.TestCase):
    def test_plist_runs_project_worker_on_network(self) -> None:
        payload = build_plist(Path("/tmp/lusas"), "/tmp/python")
        self.assertEqual(payload["Label"], SERVICE_LABEL)
        self.assertTrue(payload["RunAtLoad"])
        self.assertEqual(payload["KeepAlive"], {"NetworkState": True})
        self.assertEqual(payload["ProgramArguments"][0], "/tmp/python")


if __name__ == "__main__":
    unittest.main()
