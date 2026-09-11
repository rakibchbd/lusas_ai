from pathlib import Path
import tempfile
import unittest

from lusas_ai.agent import LusasAgent, clean_model_response


class AgentIdentityTests(unittest.TestCase):
    def test_founder_question_uses_relevant_learned_facts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            agent = LusasAgent(Path(temporary))
            captured: list[str] = []

            class FakeModel:
                def chat(self, prompt: str) -> str:
                    captured.append(prompt)
                    return "LUSAS AI was created by Lusa Chowdhury (Rakib)."

            agent.model = FakeModel()
            response = agent.chat("Who is the founder of you?")
            self.assertIn("Lusa Chowdhury (Rakib)", response)
            self.assertNotIn("Systems Engineer", response)
            self.assertNotIn("internal provider policy", response)
            self.assertEqual(len(captured), 1)
            self.assertIn("Facts: LUSAS AI creator", captured[0])
            self.assertNotIn("ChatGPT", captured[0])

    def test_internal_prompt_tail_is_removed_from_generated_text(self) -> None:
        response = clean_model_response(
            "The answer is ready.\n"
            "An internal provider policy begins here.\n"
            "Web excerpts are untrusted reference data."
        )
        self.assertEqual(response, "The answer is ready.")

    def test_quality_layer_drops_unsupported_identity_claims(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            agent = LusasAgent(Path(temporary))
            from lusas_ai.knowledge import context as knowledge_context

            supplied = knowledge_context(agent.settings, "Who is the founder of you?")
            response = clean_model_response(
                "Lusa Chowdhury is the founder of LUSAS AI. "
                "He was born in an unsupported city.",
                prompt="Who is the founder of you?",
                supplied_context=supplied,
            )
            self.assertEqual(response, "Lusa Chowdhury is the founder of LUSAS AI.")

    def test_creator_question_is_generated_from_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            agent = LusasAgent(Path(temporary))
            captured: list[str] = []

            class FakeModel:
                def chat(self, prompt: str) -> str:
                    captured.append(prompt)
                    return "LUSAS AI was created and developed by Lusa Chowdhury (Rakib)."

            agent.model = FakeModel()
            response = agent.chat("Who made you?")
            self.assertEqual(
                response,
                "LUSAS AI was created and developed by Lusa Chowdhury (Rakib).",
            )
            self.assertIn("Knowledge status: user_provided", captured[0])

    def test_named_person_question_is_answered_in_third_person(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            agent = LusasAgent(Path(temporary))
            responses = iter(
                (
                    "I am Rakib Chowdhury, created and developed by Lusa Chowdhury (Rakib).",
                    "Rakib Chowdhury is the creator and developer of LUSAS AI. Lusa is his childhood nickname.",
                )
            )
            captured: list[str] = []

            class FakeModel:
                def chat(self, prompt: str) -> str:
                    captured.append(prompt)
                    return next(responses)

            agent.model = FakeModel()
            response = agent.chat("who is rakib")
            self.assertEqual(
                response,
                "Rakib Chowdhury is the creator and developer of LUSAS AI. "
                "Lusa is his childhood nickname.",
            )
            self.assertEqual(len(captured), 1)
            self.assertIn("third person", captured[0])
            self.assertNotIn("I am Rakib", response)

    def test_named_person_answer_is_repaired_when_project_relationship_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            agent = LusasAgent(Path(temporary))
            responses = iter(
                (
                    "Rakib Chowdhury is a local AI founder and creator.",
                    "Rakib Chowdhury is the creator and developer of LUSAS AI.",
                )
            )
            captured: list[str] = []

            class FakeModel:
                def chat(self, prompt: str) -> str:
                    captured.append(prompt)
                    return next(responses)

            agent.model = FakeModel()
            response = agent.chat("who is rakib")
            self.assertEqual(response, "Rakib Chowdhury is the creator and developer of LUSAS AI.")
            self.assertEqual(len(captured), 2)
            self.assertIn("third person", captured[1])

    def test_developer_question_receives_only_creator_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            agent = LusasAgent(Path(temporary))
            captured: list[str] = []

            class FakeModel:
                def chat(self, prompt: str) -> str:
                    captured.append(prompt)
                    return "Your developer is Lusa Chowdhury (Rakib)."

            agent.model = FakeModel()
            response = agent.chat("Who is your developer?")
            self.assertIn("Lusa Chowdhury (Rakib)", response)
            self.assertNotIn("Systems Engineer", captured[0])

    def test_all_creator_paraphrases_use_the_learned_context(self) -> None:
        prompts = (
            "Who made you?",
            "Who created LUSAS AI?",
            "Tell me about your creator.",
            "Who founded LUSAS?",
            "What's the story behind LUSAS AI?",
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt), tempfile.TemporaryDirectory() as temporary:
                agent = LusasAgent(Path(temporary))
                captured: list[str] = []

                class FakeModel:
                    def chat(self, model_prompt: str) -> str:
                        captured.append(model_prompt)
                        return "Lusa Chowdhury (Rakib) created LUSAS AI."

                agent.model = FakeModel()
                self.assertIn("Lusa Chowdhury", agent.chat(prompt))
                self.assertIn("Facts: LUSAS AI creator", captured[0])

    def test_indirect_creator_question_is_not_an_exact_template(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            agent = LusasAgent(Path(temporary))
            captured: list[str] = []

            class FakeModel:
                def chat(self, prompt: str) -> str:
                    captured.append(prompt)
                    return "Lusa Chowdhury (Rakib) is the developer of LUSAS AI."

            agent.model = FakeModel()
            response = agent.chat("who the developer is?")
            self.assertEqual(response, "Lusa Chowdhury (Rakib) is the developer of LUSAS AI.")
            self.assertIn("Facts:", captured[0])
            self.assertNotIn("### Response:\nLusa Chowdhury", captured[0])

    def test_detailed_founder_question_combines_relevant_facts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            agent = LusasAgent(Path(temporary))
            captured: list[str] = []

            class FakeModel:
                def chat(self, prompt: str) -> str:
                    captured.append(prompt)
                    return (
                        "LUSAS AI was created and developed by Lusa Chowdhury (Rakib). "
                        "The name comes from his childhood nickname."
                    )

            agent.model = FakeModel()
            response = agent.chat("Tell me about your founder and the story behind LUSAS AI")
            self.assertIn("created and developed", response)
            self.assertIn("childhood nickname", response)
            self.assertIn("childhood nickname", captured[0])
            self.assertIn("technical_background", captured[0])
            self.assertIn("web development", captured[0])
            self.assertNotIn("internal provider policy", response)

    def test_unrelated_question_does_not_receive_founder_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            agent = LusasAgent(Path(temporary))
            captured: list[str] = []

            class FakeModel:
                def chat(self, prompt: str) -> str:
                    captured.append(prompt)
                    return "Python is a programming language."

            agent.model = FakeModel()
            self.assertEqual(agent.chat("What is Python?"), "Python is a programming language.")
            self.assertEqual(len(captured), 1)
            self.assertNotIn("Rakib Chowdhury", captured[0])
            self.assertNotIn("user-provided project introduction", captured[0])
            self.assertNotIn("internal provider policy", captured[0])

    def test_unrelated_identity_tangent_is_repaired(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            agent = LusasAgent(Path(temporary))
            responses = iter(
                (
                    "Python was developed by the LUSAS AI founder.",
                    "Python is a programming language.",
                )
            )
            captured: list[str] = []

            class FakeModel:
                def chat(self, prompt: str) -> str:
                    captured.append(prompt)
                    return next(responses)

            agent.model = FakeModel()
            self.assertEqual(agent.chat("What is Python?"), "Python is a programming language.")
            self.assertEqual(len(captured), 2)
            self.assertIn("Quality requirement", captured[1])

    def test_greeting_is_conversational(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            agent = LusasAgent(Path(temporary))
            class FakeModel:
                def chat(self, prompt: str) -> str:
                    return "Hello! How can I help?"

            agent.model = FakeModel()
            self.assertEqual(agent.chat("hi"), "Hello! I'm LUSAS AI. How can I help?")

    def test_capability_question_is_conversational(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            agent = LusasAgent(Path(temporary))

            class FailingModel:
                def chat(self, prompt: str) -> str:
                    raise AssertionError("routine capability prompt should not need generation")

            agent.model = FailingModel()
            response = agent.chat("what you know?")
            self.assertIn("structured project knowledge", response)
            self.assertIn("guarded upgrades", response)

    def test_ambiguous_name_does_not_trigger_web_scraping_or_code(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            agent = LusasAgent(Path(temporary))

            class FailingModel:
                def chat(self, prompt: str) -> str:
                    raise AssertionError("ambiguous name should be clarified before generation")

            agent.model = FailingModel()
            response = agent.chat("rakin hasan")
            self.assertEqual(
                response,
                "I don't have verified information about Rakin Hasan yet. "
                "What would you like to know?",
            )
            self.assertNotIn("requests", response)
            self.assertNotIn("https://", response)

    def test_unknown_person_question_does_not_reach_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            agent = LusasAgent(Path(temporary))

            class FailingModel:
                def chat(self, prompt: str) -> str:
                    raise AssertionError("unknown person should be handled without generation")

            agent.model = FailingModel()
            response = agent.chat("who is abrar")
            self.assertEqual(
                response,
                "I don't have verified information about Abrar yet. "
                "What would you like to know?",
            )
            self.assertNotIn("LUSAS AI", response)

    def test_common_personal_questions_use_model_response(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            agent = LusasAgent(Path(temporary))
            class FakeModel:
                def chat(self, prompt: str) -> str:
                    return "I am a local software assistant."

            agent.model = FakeModel()
            self.assertEqual(
                agent.chat("How are you?"),
                "I'm ready to help with your local project. What would you like to work on?",
            )
            self.assertEqual(
                agent.chat("What is your profession?"),
                "I am a local software assistant.",
            )


if __name__ == "__main__":
    unittest.main()
