import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lusas_ai.config import Settings
from lusas_ai.web_learning import MAX_CONTEXT_CHARS, context, refresh


class WebLearningTests(unittest.TestCase):
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
            self.assertEqual(stored["title"], "Python release")
            self.assertEqual(stored["verification_status"], "unverified")
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
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            result = context(settings, "What changed in transformers?")
            self.assertIn("Transformers release", result)
            self.assertIn("added a feature", result)
            self.assertIn("unverified", result)

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
