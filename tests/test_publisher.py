import unittest

from lusas_ai.publisher import PublishError


class PublisherTests(unittest.TestCase):
    def test_publish_error_is_explicit(self) -> None:
        self.assertTrue(issubclass(PublishError, RuntimeError))


if __name__ == "__main__":
    unittest.main()
