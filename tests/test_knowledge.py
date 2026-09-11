from pathlib import Path
import tempfile
import unittest

from lusas_ai.config import Settings
from lusas_ai.knowledge import ensure_seed, ingest_learning, records, retrieve


class KnowledgeTests(unittest.TestCase):
    def test_founder_introduction_is_stored_as_structured_facts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            settings = Settings(root=Path(temporary))
            seeded = ensure_seed(settings)
            founder = next(item for item in seeded if item["knowledge_id"] == "lusas-founder")
            self.assertEqual(founder["kind"], "fact")
            self.assertEqual(founder["facts"][0]["relation"], "creator")
            self.assertEqual(founder["scope"], "personal_identity")
            self.assertIsNone(founder["embeddings"])
            self.assertEqual(
                retrieve(settings, "Who founded LUSAS?")[0]["knowledge_id"],
                "lusas-founder",
            )
            self.assertFalse(
                any(item["scope"] == "personal_identity" for item in retrieve(settings, "What is Python?"))
            )

    def test_learning_is_incremental_and_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            settings = Settings(root=Path(temporary))
            first = ingest_learning(settings, "What is a decorator?", "A decorator wraps a function.")
            second = ingest_learning(settings, "What is a decorator?", "A decorator wraps a function.")
            stored = records(settings)
            self.assertEqual(first["knowledge_id"], second["knowledge_id"])
            self.assertEqual(len(stored), 1)
            self.assertEqual(stored[0]["kind"], "learned_fact")
            self.assertEqual(stored[0]["source_type"], "approved_learning")
            self.assertIn("decorator", stored[0]["index_terms"])


if __name__ == "__main__":
    unittest.main()
