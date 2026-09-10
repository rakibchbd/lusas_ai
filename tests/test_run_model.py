import unittest

from lusas_ai.identity import IDENTITY_RESPONSE
from training.run_model import generate_loaded


class TrainedModelIdentityTests(unittest.TestCase):
    def test_creator_question_never_reaches_the_model(self) -> None:
        self.assertEqual(
            generate_loaded(None, None, None, "cpu", "who is your developer?", 128),
            IDENTITY_RESPONSE,
        )

    def test_indirect_creator_question_never_reaches_the_model(self) -> None:
        self.assertEqual(
            generate_loaded(None, None, None, "cpu", "who the maker is?", 128),
            IDENTITY_RESPONSE,
        )


if __name__ == "__main__":
    unittest.main()
