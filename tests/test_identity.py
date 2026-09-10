import unittest

from lusas_ai.identity import GREETING_QUESTION


class GreetingTests(unittest.TestCase):
    def test_common_greetings_match(self) -> None:
        for prompt in ("hi", "Hello!", "good morning"):
            with self.subTest(prompt=prompt):
                self.assertTrue(GREETING_QUESTION.fullmatch(prompt))

    def test_greeting_pattern_does_not_match_normal_requests(self) -> None:
        self.assertFalse(GREETING_QUESTION.fullmatch("help me write Python"))


if __name__ == "__main__":
    unittest.main()
