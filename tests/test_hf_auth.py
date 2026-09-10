import os
import unittest
from unittest.mock import patch

from training.hf_auth import auth_kwargs


class HuggingFaceAuthTests(unittest.TestCase):
    def test_auth_is_empty_when_token_is_not_configured(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(auth_kwargs(), {})

    def test_auth_reads_token_only_from_environment(self) -> None:
        with patch.dict(os.environ, {"HF_TOKEN": "hf_test"}, clear=True):
            self.assertEqual(auth_kwargs(), {"token": "hf_test"})

    def test_auth_ignores_blank_token(self) -> None:
        with patch.dict(os.environ, {"HF_TOKEN": "  "}, clear=True):
            self.assertEqual(auth_kwargs(), {})


if __name__ == "__main__":
    unittest.main()
