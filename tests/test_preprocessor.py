import unittest
from unittest.mock import patch

from backend.preprocessor import TextPreprocessor


class TextPreprocessorTests(unittest.TestCase):
    def setUp(self):
        self.preprocessor = TextPreprocessor(use_lemmatization=False, min_token_length=2)

    def test_clean_removes_placeholders_urls_numbers_and_punctuation(self):
        text = "Hello {{Order Number}} please visit https://example.com/order/123 ASAP!"
        cleaned = self.preprocessor.clean(text)

        self.assertEqual(cleaned, "hello please visit asap")

    def test_remove_stopwords_and_length_filter(self):
        tokens = ["refund", "my", "order", "a", "ok"]
        filtered = self.preprocessor.remove_stopwords(tokens)

        self.assertEqual(filtered, ["refund", "order", "ok"])

    def test_preprocess_pipeline_with_mocked_tokenizer(self):
        with patch.object(self.preprocessor, "tokenize", return_value=["Need", "refund", "for", "order", "issue"]):
            tokens = self.preprocessor.preprocess("Need refund for order 123")

        self.assertIsInstance(tokens, list)
        self.assertIn("refund", tokens)
        self.assertIn("order", tokens)
        self.assertNotIn("for", tokens)
        self.assertNotIn("issue", tokens)

    def test_clean_non_string_returns_empty_string(self):
        self.assertEqual(self.preprocessor.clean(None), "")
        self.assertEqual(self.preprocessor.clean(123), "")


if __name__ == "__main__":
    unittest.main()
