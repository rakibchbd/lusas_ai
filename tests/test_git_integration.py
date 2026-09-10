from pathlib import Path
import subprocess
import tempfile
import unittest

from lusas_ai.git_integration import GitIntegration


class GitIntegrationTests(unittest.TestCase):
    def test_commit_is_local_and_limited_to_applied_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "lusas_ai" / "value.py"
            source.parent.mkdir()
            source.write_text("VALUE = 1\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.invalid"],
                cwd=root,
                check=True,
            )
            subprocess.run(["git", "add", "lusas_ai/value.py"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "baseline"], cwd=root, check=True)
            source.write_text("VALUE = 2\n", encoding="utf-8")

            result = GitIntegration().commit_applied(
                root, ("lusas_ai/value.py",), "improve value"
            )

            self.assertTrue(result.committed)
            self.assertEqual(
                subprocess.run(
                    ["git", "show", "--format=%s", "--no-patch", "HEAD"],
                    cwd=root,
                    text=True,
                    capture_output=True,
                    check=True,
                ).stdout.strip(),
                "LUSAS autonomous upgrade: improve value",
            )

    def test_existing_index_changes_block_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "lusas_ai" / "value.py"
            source.parent.mkdir()
            source.write_text("VALUE = 1\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.invalid"],
                cwd=root,
                check=True,
            )
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "baseline"], cwd=root, check=True)
            source.write_text("VALUE = 2\n", encoding="utf-8")
            subprocess.run(["git", "add", str(source)], cwd=root, check=True)

            result = GitIntegration().commit_applied(
                root, ("lusas_ai/value.py",), "blocked"
            )

            self.assertFalse(result.committed)
            self.assertIn("staged changes", result.error or "")


if __name__ == "__main__":
    unittest.main()
