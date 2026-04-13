import unittest

from backend.evaluator import (
    KFoldsCrossValidator,
    MetricsCalculator,
)
from backend.naive_bayes import NaiveBayesClassifier


class MetricsCalculatorTests(unittest.TestCase):
    def test_accuracy_and_invalid_lengths(self):
        self.assertAlmostEqual(
            MetricsCalculator.accuracy(["A", "B", "A"], ["A", "C", "A"]),
            2 / 3,
        )
        with self.assertRaises(ValueError):
            MetricsCalculator.accuracy(["A"], [])
        with self.assertRaises(ValueError):
            MetricsCalculator.accuracy([], [])

    def test_confusion_matrix_and_class_metrics(self):
        classes = ["A", "B", "C"]
        y_true = ["A", "B", "A", "C"]
        y_pred = ["A", "A", "C", "C"]

        matrix = MetricsCalculator.confusion_matrix(y_true, y_pred, classes)
        per_class = MetricsCalculator.precision_recall_f1_per_class(y_true, y_pred, classes)
        macro_f1 = MetricsCalculator.macro_f1(per_class)

        self.assertEqual(matrix, [[1, 0, 1], [1, 0, 0], [0, 0, 1]])
        self.assertEqual(per_class["A"]["tp"], 1)
        self.assertEqual(per_class["B"]["recall"], 0.0)
        self.assertGreaterEqual(macro_f1, 0.0)
        self.assertLessEqual(macro_f1, 1.0)


class KFoldsCrossValidatorTests(unittest.TestCase):
    def test_invalid_k_raises(self):
        with self.assertRaises(ValueError):
            KFoldsCrossValidator(k=1)

    def test_get_folds_covers_all_samples(self):
        cv = KFoldsCrossValidator(k=3, shuffle=False)
        folds = cv.get_folds(6)

        self.assertEqual(len(folds), 3)
        val_indices = sorted(index for _, val in folds for index in val)
        self.assertEqual(val_indices, list(range(6)))

    def test_run_with_real_model(self):
        X = [
            ["refund", "order"],
            ["billing", "invoice"],
            ["refund", "money"],
            ["account", "password"],
            ["refund", "charge"],
            ["billing", "payment"],
        ]
        y = ["REFUND", "BILLING", "REFUND", "ACCOUNT", "REFUND", "BILLING"]
        classes = ["ACCOUNT", "BILLING", "REFUND"]

        def train_fn(X_tr, y_tr):
            model = NaiveBayesClassifier(alpha=1.0, vocab_min_freq=1)
            model.train(X_tr, y_tr)
            return model

        def predict_fn(model, X_val):
            return model.predict_batch(X_val)

        cv = KFoldsCrossValidator(k=3, shuffle=True, random_seed=7)
        results = cv.run(X, y, train_fn, predict_fn, classes, verbose=False)

        self.assertEqual(results["k"], 3)
        self.assertEqual(len(results["fold_results"]), 3)
        self.assertIn("mean_accuracy", results)
        self.assertIn("mean_macro_f1", results)


if __name__ == "__main__":
    unittest.main()
