import unittest

from backend.naive_bayes import NaiveBayesClassifier
from backend.vocabulary import Vocabulary


class VocabularyTests(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            ["refund", "order", "refund"],
            ["billing", "invoice"],
            ["account", "reset", "password"],
        ]

    def test_fit_transform_and_lookup(self):
        vocab = Vocabulary(min_freq=1).fit(self.corpus)
        bow = vocab.transform(["refund", "refund", "missing"])

        self.assertTrue(vocab.contains("refund"))
        self.assertGreaterEqual(vocab.size, 3)
        self.assertEqual(bow["refund"], 2)
        self.assertNotIn("missing", bow)

    def test_transform_before_fit_raises(self):
        vocab = Vocabulary(min_freq=1)
        with self.assertRaises(RuntimeError):
            vocab.transform(["refund"])


class NaiveBayesClassifierTests(unittest.TestCase):
    def setUp(self):
        self.X = [
            ["refund", "order"],
            ["billing", "invoice"],
            ["refund", "money"],
            ["account", "password"],
        ]
        self.y = ["REFUND", "BILLING", "REFUND", "ACCOUNT"]

    def test_invalid_alpha_raises(self):
        with self.assertRaises(ValueError):
            NaiveBayesClassifier(alpha=0)

    def test_train_with_mismatched_lengths_raises(self):
        model = NaiveBayesClassifier(alpha=1.0, vocab_min_freq=1)
        with self.assertRaises(ValueError):
            model.train(self.X, self.y[:-1])

    def test_train_with_empty_corpus_raises(self):
        model = NaiveBayesClassifier(alpha=1.0, vocab_min_freq=1)
        with self.assertRaises(ValueError):
            model.train([], [])

    def test_predict_before_training_raises(self):
        model = NaiveBayesClassifier(alpha=1.0, vocab_min_freq=1)
        with self.assertRaises(RuntimeError):
            model.predict(["refund"])

    def test_train_predict_and_predict_proba(self):
        model = NaiveBayesClassifier(alpha=1.0, vocab_min_freq=1)
        model.train(self.X, self.y)

        prediction = model.predict(["refund", "money"])
        probabilities = model.predict_proba(["refund", "money"])

        self.assertEqual(prediction, "REFUND")
        self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=6)
        self.assertEqual(max(probabilities, key=probabilities.get), prediction)

    def test_serialization_roundtrip(self):
        model = NaiveBayesClassifier(alpha=1.0, vocab_min_freq=1)
        model.train(self.X, self.y)

        data = model.to_dict()
        restored = NaiveBayesClassifier.from_dict(data)

        self.assertEqual(restored.classes_, model.classes_)
        self.assertEqual(restored.predict(["refund", "money"]), "REFUND")


if __name__ == "__main__":
    unittest.main()
