import unittest

from training.run_model import generate_loaded, strip_internal_prompt_leak


class TrainedModelIdentityTests(unittest.TestCase):
    class FakeTensor:
        shape = (1, 1)

        def to(self, device: str) -> "TrainedModelIdentityTests.FakeTensor":
            return self

        def __getitem__(self, key: object) -> "TrainedModelIdentityTests.FakeTensor":
            return self

    class FakeTokenizer:
        def __init__(self) -> None:
            self.formatted = ""

        def __call__(self, formatted: str, **kwargs: object) -> dict[str, object]:
            self.formatted = formatted
            return {"input_ids": TrainedModelIdentityTests.FakeTensor()}

        def decode(self, tokens: object, skip_special_tokens: bool = True) -> str:
            return "Lusa Chowdhury (Rakib) created LUSAS AI."

    class FakeTorch:
        class NoGrad:
            def __enter__(self) -> None:
                return None

            def __exit__(self, *args: object) -> None:
                return None

        def no_grad(self) -> "TrainedModelIdentityTests.NoGrad":
            return TrainedModelIdentityTests.FakeTorch.NoGrad()

        def manual_seed(self, seed: int) -> None:
            return None

    class FakeModel:
        def generate(self, **kwargs: object) -> list[object]:
            return [TrainedModelIdentityTests.FakeTensor()]

    def _generate(self, prompt: str) -> tuple[str, str]:
        tokenizer = self.FakeTokenizer()
        response = generate_loaded(
            self.FakeTorch(),
            tokenizer,
            self.FakeModel(),
            "cpu",
            prompt,
            128,
        )
        return response, tokenizer.formatted

    def test_generated_prompt_tail_is_removed(self) -> None:
        self.assertEqual(
            strip_internal_prompt_leak(
                "Useful answer.\n"
                "An internal provider policy begins here.\n"
                "Do not access credentials."
            ),
            "Useful answer.",
        )

    def test_creator_question_reaches_model_with_relevant_context(self) -> None:
        response, formatted = self._generate("who is your developer?")
        self.assertEqual(response, "Lusa Chowdhury (Rakib) created LUSAS AI.")
        self.assertIn("Facts: LUSAS AI creator", formatted)
        self.assertIn("Lusa Chowdhury (Rakib)", formatted)
        self.assertNotIn("ChatGPT", formatted)

    def test_named_person_question_does_not_return_first_person_identity(self) -> None:
        tokenizer = self.FakeTokenizer()

        def decode(_: object, skip_special_tokens: bool = True) -> str:
            return "I am Rakib Chowdhury, created and developed by Lusa Chowdhury (Rakib)."

        tokenizer.decode = decode  # type: ignore[method-assign]
        response = generate_loaded(
            self.FakeTorch(),
            tokenizer,
            self.FakeModel(),
            "cpu",
            "who is rakib",
            128,
        )
        self.assertIn("Rakib Chowdhury", response)
        self.assertNotIn("I am Rakib", response)

    def test_indirect_creator_question_gets_context_without_fixed_answer(self) -> None:
        response, formatted = self._generate("who the maker is?")
        self.assertEqual(response, "Lusa Chowdhury (Rakib) created LUSAS AI.")
        self.assertIn("Knowledge status: user_provided", formatted)
        self.assertNotIn("### Response:\nLUSAS AI was created", formatted)

    def test_unrelated_question_does_not_get_personal_context(self) -> None:
        _, formatted = self._generate("What is Python?")
        self.assertNotIn("Lusa Chowdhury", formatted)
        self.assertNotIn("Facts: LUSAS AI creator", formatted)

    def test_ambiguous_name_is_clarified_without_generation(self) -> None:
        tokenizer = self.FakeTokenizer()
        response = generate_loaded(
            self.FakeTorch(),
            tokenizer,
            self.FakeModel(),
            "cpu",
            "rakin hasan",
            128,
        )
        self.assertEqual(
            response,
            "I don't have verified information about Rakin Hasan yet. "
            "What would you like to know?",
        )
        self.assertEqual(tokenizer.formatted, "")

    def test_unknown_person_question_is_clarified_without_generation(self) -> None:
        tokenizer = self.FakeTokenizer()
        response = generate_loaded(
            self.FakeTorch(),
            tokenizer,
            self.FakeModel(),
            "cpu",
            "who is abrar",
            128,
        )
        self.assertEqual(
            response,
            "I don't have verified information about Abrar yet. "
            "What would you like to know?",
        )
        self.assertEqual(tokenizer.formatted, "")

    def test_routine_conversation_is_answered_without_generation(self) -> None:
        tokenizer = self.FakeTokenizer()
        response = generate_loaded(
            self.FakeTorch(),
            tokenizer,
            self.FakeModel(),
            "cpu",
            "Hello",
            128,
        )
        self.assertEqual(response, "Hello! I'm LUSAS AI. How can I help?")
        self.assertEqual(tokenizer.formatted, "")

    def test_parameter_question_is_answered_without_generation(self) -> None:
        tokenizer = self.FakeTokenizer()
        response = generate_loaded(
            self.FakeTorch(),
            tokenizer,
            self.FakeModel(),
            "cpu",
            "how many parameters do you have?",
            128,
        )
        self.assertIn("0.5 billion parameters", response)
        self.assertEqual(tokenizer.formatted, "")


if __name__ == "__main__":
    unittest.main()
