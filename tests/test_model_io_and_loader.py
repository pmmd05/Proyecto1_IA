import csv
import tempfile
import unittest
from pathlib import Path
from urllib.error import URLError
from unittest.mock import patch

from backend.data_loader import DataLoader
from backend.model_io import ModelManager
from backend.naive_bayes import NaiveBayesClassifier


class ModelManagerTests(unittest.TestCase):
    def setUp(self):
        self.X = [["refund", "order"], ["billing", "invoice"]]
        self.y = ["REFUND", "BILLING"]
        self.model = NaiveBayesClassifier(alpha=1.0, vocab_min_freq=1)
        self.model.train(self.X, self.y)

    def test_save_and_load_pickle_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelManager(models_dir=tmpdir)
            save_path = manager.save_pickle(self.model, filename="model.pkl", metadata={"note": "test"})
            loaded_model, metadata = manager.load_pickle("model.pkl")

            self.assertTrue(Path(save_path).exists())
            self.assertEqual(metadata["note"], "test")
            self.assertEqual(loaded_model.predict(["refund"]), "REFUND")

    def test_save_and_load_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelManager(models_dir=tmpdir)
            save_path = manager.save_json(self.model, filename="model.json", metadata={"note": "test"})
            loaded_model, metadata = manager.load_json("model.json")

            self.assertTrue(Path(save_path).exists())
            self.assertEqual(metadata["note"], "test")
            self.assertEqual(loaded_model.predict(["billing"]), "BILLING")

    def test_load_missing_model_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelManager(models_dir=tmpdir)
            with self.assertRaises(FileNotFoundError):
                manager.load()


class DataLoaderTests(unittest.TestCase):
    def test_load_from_csv_and_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            csv_path = data_dir / "sample.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["instruction", "category"])
                writer.writeheader()
                writer.writerow({"instruction": "  I need a refund  ", "category": "refund"})
                writer.writerow({"instruction": "", "category": "ignored"})

            loader = DataLoader(data_dir=data_dir)
            texts, labels = loader.load_from_csv(filename="sample.csv")

            self.assertEqual(texts, ["I need a refund"])
            self.assertEqual(labels, ["REFUND"])

            with self.assertRaises(FileNotFoundError):
                loader.load_from_csv(filename="missing.csv")

    def test_load_from_huggingface_network_failure_is_wrapped(self):
        class FailingResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                raise URLError("offline")

        with tempfile.TemporaryDirectory() as tmpdir:
            loader = DataLoader(data_dir=tmpdir)
            with patch("backend.data_loader.urlopen", return_value=FailingResponse()):
                with self.assertRaises(RuntimeError):
                    loader.load_from_huggingface(save_local=False)


if __name__ == "__main__":
    unittest.main()
