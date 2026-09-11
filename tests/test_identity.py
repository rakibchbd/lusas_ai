import unittest

from lusas_ai.identity import CREATOR_QUESTION, is_creator_question


class IdentityIntentTests(unittest.TestCase):
    def test_creator_pattern_matches_founder_paraphrases(self) -> None:
        prompts = (
            "Who is the founder of you?",
            "Who made you?",
            "Who created LUSAS AI?",
            "Tell me about your creator",
            "Who founded LUSAS?",
            "What's the story behind LUSAS AI?",
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertTrue(CREATOR_QUESTION.search(prompt))
                self.assertTrue(is_creator_question(prompt))

    def test_unrelated_questions_are_not_creator_questions(self) -> None:
        for prompt in ("What is Python?", "Explain recursion", "Write a CLI tool"):
            with self.subTest(prompt=prompt):
                self.assertFalse(is_creator_question(prompt))


if __name__ == "__main__":
    unittest.main()
