from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest

from lusas_ai.audit import run as run_audit
from lusas_ai.config import Settings
from lusas_ai.control import (
    automatic_deploy_allowed,
    code_evolution_allowed,
    model_learning_allowed,
    read as read_control,
    update as update_control,
    web_research_allowed,
)
from lusas_ai.engine import EvolutionEngine, read_cycles
from lusas_ai.gaps import detect, record
from lusas_ai.knowledge import graph, is_outdated, upsert_web
from lusas_ai.scorecard import measure, record as record_scorecard
from lusas_ai.skills import ensure_builtin, record_use


class AutonomyComponentTests(unittest.TestCase):
    def test_runtime_controls_bound_every_automatic_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = Settings(root=root, autonomy_level=5)
            state = read_control(root, settings.autonomy_level)
            self.assertTrue(automatic_deploy_allowed(state))
            self.assertTrue(code_evolution_allowed(state))
            self.assertTrue(model_learning_allowed(state))
            self.assertTrue(web_research_allowed(state))
            state = update_control(
                root,
                5,
                autonomy_level=1,
                emergency_stop=True,
                auto_deploy_enabled=True,
            )
            self.assertFalse(automatic_deploy_allowed(state))
            self.assertFalse(code_evolution_allowed(state))
            self.assertFalse(model_learning_allowed(state))
            self.assertFalse(web_research_allowed(state))
            self.assertEqual(
                update_control(root, 2, autonomy_level=5)["autonomy_level"], 2
            )

    def test_knowledge_graph_preserves_source_and_freshness_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            settings = Settings(root=Path(temporary))
            article = {
                "id": "source-1",
                "title": "Python programming update",
                "url": "https://example.com/update",
                "summary": "A programming update for Python.",
                "source": "https://example.com/feed.xml",
                "fetched_at": "2026-01-01T00:00:00+00:00",
                "verification_status": "corroborated",
            }
            self.assertEqual(upsert_web(settings, [article]), 1)
            stored = json.loads(settings.knowledge_path.read_text().splitlines()[0])
            self.assertEqual(stored["source"], article["url"])
            self.assertEqual(stored["verification"], "corroborated")
            self.assertIn("programming", stored["domain"])
            self.assertFalse(is_outdated(stored, datetime.now(timezone.utc)))
            stale = dict(stored)
            stale["expires_at"] = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
            self.assertTrue(is_outdated(stale))
            knowledge_graph = graph(settings)
            self.assertEqual(len(knowledge_graph["nodes"]), 1)
            self.assertTrue(knowledge_graph["edges"])

    def test_audit_gap_skill_scorecard_and_cycle_are_real_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = Settings(root=root)
            (root / "lusas_ai").mkdir()
            (root / "lusas_ai" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
            audit = run_audit(settings, evolution_id="EV-000001")
            self.assertEqual(audit["status"], "findings")
            gap = record(settings, "testing", "The audit found a missing required file.")
            self.assertEqual(gap["status"], "open")
            self.assertTrue(detect(settings, audit))
            skills = ensure_builtin(settings)
            self.assertGreaterEqual(len(skills), 5)
            used = record_use(settings, "local_chat", succeeded=True, benchmark_score=0.9)
            self.assertEqual(used["success_rate"], 1.0)
            scorecard = measure(settings, audit_result={"status": "healthy"})
            self.assertIn("intelligence", scorecard["dimensions"])
            self.assertIsNone(scorecard["dimensions"]["intelligence"]["score"])
            record_scorecard(settings, version="v0.000", audit_result=audit)
            engine = EvolutionEngine(settings)
            cycle = engine.begin()
            engine.phase(cycle, "audit", findings=0)
            engine.complete(cycle, status="rejected", version="v0.000")
            self.assertEqual(read_cycles(settings)[0]["evolution_id"], cycle.evolution_id)


if __name__ == "__main__":
    unittest.main()
