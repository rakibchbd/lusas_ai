from pathlib import Path
import tempfile
import unittest

from lusas_ai.agent import LusasAgent
from lusas_ai.identity import GREETING_RESPONSE, IDENTITY_RESPONSE


class AgentIdentityTests(unittest.TestCase):
    def test_creator_question_uses_lusa_identity_without_model_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            agent = LusasAgent(Path(temporary))
            self.assertEqual(
                agent.chat("Who made you?"),
                IDENTITY_RESPONSE,
            )

    def test_developer_question_returns_complete_creator_biography(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            agent = LusasAgent(Path(temporary))
            response = agent.chat("Who is your developer?")
            self.assertEqual(response, IDENTITY_RESPONSE)
            self.assertIn("Systems Engineer", response)

    def test_indirect_creator_question_returns_one_biography(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            agent = LusasAgent(Path(temporary))
            response = agent.chat("who the developer is?")
            self.assertEqual(response, IDENTITY_RESPONSE)
            self.assertNotIn("### Instruction:", response)

    def test_greeting_is_conversational_without_model_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            agent = LusasAgent(Path(temporary))
            self.assertEqual(agent.chat("hi"), GREETING_RESPONSE)

    def test_common_personal_questions_have_short_answers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            agent = LusasAgent(Path(temporary))
            self.assertIn("human age", agent.chat("How old are you?"))
            self.assertIn("working well", agent.chat("How are you?"))
            self.assertIn("local AI assistant", agent.chat("What is your profession?"))


if __name__ == "__main__":
    unittest.main()
