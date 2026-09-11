import unittest

from lusas_ai.conversation import quick_response


class ConversationTests(unittest.TestCase):
    def test_greetings(self) -> None:
        self.assertIsNotNone(quick_response("Hi"))
        self.assertIsNotNone(quick_response("good morning"))

    def test_capability_variants(self) -> None:
        for prompt in ("what you know?", "What can you do?", "Tell me about yourself"):
            with self.subTest(prompt=prompt):
                self.assertIn("local knowledge base", quick_response(prompt) or "")

    def test_user_identity_and_parameter_questions(self) -> None:
        self.assertIn("Rakib Chowdhury", quick_response("do you know me?") or "")
        self.assertIn("0.5 billion parameters", quick_response("how many peremeter do you have?") or "")

    def test_specific_questions_still_use_the_model(self) -> None:
        self.assertIsNone(quick_response("What is Python?"))
        self.assertIsNone(quick_response("Write a Python function"))


if __name__ == "__main__":
    unittest.main()
