import unittest

from training.run_model import identity_response


class ModelIdentityTests(unittest.TestCase):
    def test_creator_questions_use_lusas_branding(self) -> None:
        response = identity_response("Who created you?")
        self.assertIsNotNone(response)
        self.assertIn("LUSAS AI", response)
        self.assertIn("Lusa Chowdhury (Rakib)", response)

    def test_openai_identity_questions_use_lusas_branding(self) -> None:
        response = identity_response("Were you created by OpenAI?")
        self.assertIsNotNone(response)
        self.assertIn("developed by Lusa Chowdhury (Rakib)", response)

    def test_normal_coding_prompt_reaches_model(self) -> None:
        self.assertIsNone(identity_response("Write a Python function to add two numbers."))


if __name__ == "__main__":
    unittest.main()
