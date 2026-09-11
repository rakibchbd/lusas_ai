import unittest

from lusas_ai.identity import (
    CREATOR_QUESTION,
    PERSON_IDENTITY_QUESTION,
    ambiguous_name_response,
    is_ambiguous_name_prompt,
    is_creator_question,
    is_person_identity_question,
    unknown_person_response,
    unknown_person_subject,
)


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

    def test_named_person_questions_are_classified_separately(self) -> None:
        for prompt in ("Who is Rakib?", "Who is Rakib Chowdhury?", "Tell me about Lusa"):
            with self.subTest(prompt=prompt):
                self.assertTrue(PERSON_IDENTITY_QUESTION.search(prompt))
                self.assertTrue(is_person_identity_question(prompt))
                self.assertFalse(is_creator_question(prompt))

    def test_name_fragment_is_not_treated_as_a_request_to_write_code(self) -> None:
        self.assertTrue(is_ambiguous_name_prompt("rakin hasan"))
        self.assertEqual(
            ambiguous_name_response("rakin hasan"),
            "I don't have verified information about Rakin Hasan yet. What would you like to know?",
        )
        self.assertFalse(is_ambiguous_name_prompt("Write a Python function"))

    def test_unknown_person_question_is_detected_without_guessing(self) -> None:
        self.assertEqual(unknown_person_subject("who is abrar"), "abrar")
        self.assertEqual(
            unknown_person_response("who is abrar"),
            "I don't have verified information about Abrar yet. What would you like to know?",
        )
        self.assertIsNone(unknown_person_subject("Who is Python?"))


if __name__ == "__main__":
    unittest.main()
