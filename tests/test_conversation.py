import unittest

from lusas_ai.conversation import quick_response


class ConversationTests(unittest.TestCase):
    def test_greetings(self) -> None:
        self.assertIsNotNone(quick_response("Hi"))
        self.assertIsNotNone(quick_response("good morning"))
        self.assertIn("LUSAS AI", quick_response("Hey, how are you?") or "")

    def test_common_conversation_variants(self) -> None:
        cases = {
            "Thank you for your help": "welcome",
            "I don't understand": "explain",
            "Could you please repeat that?": "repeat",
            "Who are you?": "LUSAS AI",
            "Can you help me?": "tell me",
            "Are you there?": "here",
            "That makes sense.": "continue",
            "No, that's not what I meant.": "correcting",
        }
        for prompt, expected in cases.items():
            with self.subTest(prompt=prompt):
                self.assertIn(expected.lower(), (quick_response(prompt) or "").lower())

    def test_capability_variants(self) -> None:
        for prompt in ("what you know?", "What can you do?", "Tell me about yourself"):
            with self.subTest(prompt=prompt):
                self.assertIn("local knowledge base", quick_response(prompt) or "")

    def test_user_identity_and_parameter_questions(self) -> None:
        self.assertIn("Rakib Chowdhury", quick_response("do you know me?") or "")
        response = quick_response("how many peremeter do you have?") or ""
        self.assertIn("Sara 1.0", response)
        self.assertIn("Lira 1.0", response)
        self.assertNotIn("100000000", response)

    def test_specific_questions_still_use_the_model(self) -> None:
        self.assertIsNone(quick_response("What is Python?"))
        self.assertIsNone(quick_response("Write a Python function"))


if __name__ == "__main__":
    unittest.main()
