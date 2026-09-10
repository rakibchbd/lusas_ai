from pathlib import Path
import tempfile
import unittest

from lusas_ai.config import Settings
from lusas_ai.evolution import (
    CandidateWorkspace,
    EvolutionOrchestrator,
    EvolutionRejected,
    ProtectedPathValidator,
    SecurityChecker,
)


class FakeModel:
    def __init__(self, response: str):
        self.response = response

    def chat(self, prompt: str) -> str:
        return self.response


class EvolutionTests(unittest.TestCase):
    def test_validator_rejects_protected_and_traversal_paths(self) -> None:
        validator = ProtectedPathValidator()
        for path in ("config.json", "lusas_ai/../config.json", ".git/hooks/x"):
            with self.subTest(path=path), self.assertRaises(EvolutionRejected):
                validator.validate(path)

    def test_security_checker_rejects_shell_capable_imports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "lusas_ai").mkdir()
            (root / "tests").mkdir()
            candidate = CandidateWorkspace.create(
                root, {"lusas_ai/bad.py": "import subprocess\n"}
            )
            result = SecurityChecker().check(candidate)
            self.assertFalse(result.passed)
            self.assertIn("forbidden import subprocess", result.output)

    def test_orchestrator_tests_candidate_before_staging(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "lusas_ai").mkdir()
            (root / "tests").mkdir()
            (root / "lusas_ai" / "__init__.py").write_text("", encoding="utf-8")
            (root / "tests" / "test_value.py").write_text(
                "import unittest\n"
                "from lusas_ai.value import VALUE\n"
                "class ValueTests(unittest.TestCase):\n"
                "    def test_value(self): self.assertEqual(VALUE, 2)\n",
                encoding="utf-8",
            )
            model = FakeModel(
                '{"summary":"improve value","files":'
                '{"lusas_ai/value.py":"VALUE = 2\\n"}}'
            )
            settings = Settings(root=root, notify_file=".lusas/events.jsonl")
            result = EvolutionOrchestrator(settings, model).run(
                "make the value reliable", apply=False
            )
            self.assertEqual(result.status, "staged")
            self.assertIsNotNone(result.metrics)
            self.assertTrue(result.metrics.tests_passed)
            self.assertFalse((root / "lusas_ai" / "value.py").exists())
            self.assertTrue(result.candidate and result.candidate.exists())


if __name__ == "__main__":
    unittest.main()
