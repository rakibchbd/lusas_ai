from pathlib import Path
import tempfile
import unittest

from lusas_ai.cycle_state import fingerprint, read, write


class CycleStateTests(unittest.TestCase):
    def test_fingerprint_changes_when_input_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.txt"
            path.write_text("one", encoding="utf-8")
            first = fingerprint([path])
            path.write_text("two", encoding="utf-8")
            self.assertNotEqual(first, fingerprint([path]))

    def test_state_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.json"
            write(path, {"last_status": "promoted"})
            self.assertEqual(read(path)["last_status"], "promoted")


if __name__ == "__main__":
    unittest.main()
