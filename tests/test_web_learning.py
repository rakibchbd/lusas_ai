import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lusas_ai.config import Settings
from lusas_ai.control import update as update_control
from lusas_ai.web_learning import MAX_CONTEXT_CHARS, context, research_context, research_needed, refresh


class WebLearningTests(unittest.TestCase):
    def test_runtime_control_can_disable_web_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = Settings(
                root=root,
                web_learning_enabled=True,
                web_sources=("https://example.com/feed.xml",),
                web_allowed_domains=("example.com",),
            )
            update_control(root, settings.autonomy_level, web_research_enabled=False)
            with patch("lusas_ai.web_learning._fetch_source") as fetch:
                result = refresh(settings, force=True)
            self.assertEqual(result["status"], "disabled")
            fetch.assert_not_called()

    def test_refresh_stores_bounded_article_and_training_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = Settings(
                root=root,
                web_learning_enabled=True,
                web_sources=("https://example.com/feed.xml",),
                web_allowed_domains=("example.com",),
            )
            article = {
                "id": "article-1",
                "title": "Python release",
                "url": "https://example.com/python",
                "summary": "A new Python release improves diagnostics.",
                "published": "today",
                "source": "https://example.com/feed.xml",
                "fetched_at": "2026-01-01T00:00:00+00:00",
            }
            with patch("lusas_ai.web_learning._fetch_source", return_value=[article]):
                result = refresh(settings, force=True)

            self.assertEqual(result["status"], "refreshed")
            self.assertEqual(result["new_items"], 1)
            stored = json.loads(settings.web_cache_path.read_text().splitlines()[0])
            training = json.loads(settings.web_training_path.read_text().splitlines()[0])
            knowledge = json.loads(settings.knowledge_path.read_text().splitlines()[0])
            self.assertEqual(stored["title"], "Python release")
            self.assertEqual(stored["verification_status"], "unverified")
            self.assertEqual(knowledge["kind"], "web_evidence")
            self.assertIn("instruction", training)
            self.assertIn("untrusted", training["output"])

    def test_search_context_uses_cached_web_data_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = Settings(root=root, web_learning_enabled=True)
            settings.web_cache_path.parent.mkdir(parents=True, exist_ok=True)
            settings.web_cache_path.write_text(
                json.dumps(
                    {
                        "id": "article-1",
                        "title": "Transformers release",
                        "url": "https://example.com/release",
                        "summary": "The transformers library added a feature.",
                        "approval_status": "approved",
                        "verification_status": "corroborated",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            result = context(settings, "What changed in transformers?")
            self.assertIn("Transformers release", result)
            self.assertIn("added a feature", result)
            self.assertIn("Approved web reference", result)

    def test_definition_question_does_not_get_release_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = Settings(root=root, web_learning_enabled=True)
            settings.web_cache_path.parent.mkdir(parents=True, exist_ok=True)
            settings.web_cache_path.write_text(
                json.dumps(
                    {
                        "id": "article-1",
                        "title": "Python release",
                        "url": "https://example.com/python",
                        "summary": "A new Python release improves diagnostics.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(context(settings, "What is Python?"), "")

    def test_research_fallback_refreshes_allowlisted_sources_for_unknown_questions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = Settings(root=root, web_learning_enabled=True)

            def fake_refresh(_settings: Settings, force: bool = False) -> dict[str, object]:
                self.assertFalse(force)
                settings.web_cache_path.parent.mkdir(parents=True, exist_ok=True)
                settings.web_cache_path.write_text(
                    json.dumps(
                        {
                            "id": "article-1",
                            "title": "Rust language",
                            "url": "https://example.com/rust",
                            "summary": "Rust is a systems programming language.",
                            "approval_status": "approved",
                            "verification_status": "admin_approved",
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return {"status": "refreshed"}

            with patch("lusas_ai.web_learning.refresh", side_effect=fake_refresh):
                result = research_context(settings, "What is Rust?")
            self.assertIn("Rust language", result)
            self.assertIn("systems programming", result)

    def test_research_decision_skips_code_requests(self) -> None:
        self.assertTrue(research_needed("What is the latest Python release?"))
        self.assertTrue(research_needed("What is Python?"))
        self.assertFalse(research_needed("Write a Python function"))

    def test_context_is_bounded_for_the_local_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = Settings(root=root, web_learning_enabled=True)
            settings.web_cache_path.parent.mkdir(parents=True, exist_ok=True)
            settings.web_cache_path.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "id": str(index),
                            "title": "performance update",
                        "url": "https://example.com/update",
                        "summary": "x" * 3_000,
                        "approval_status": "approved",
                        "verification_status": "corroborated",
                        }
                    )
                    for index in range(3)
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertLessEqual(
                len(context(settings, "performance update")), MAX_CONTEXT_CHARS
            )


if __name__ == "__main__":
    unittest.main()
