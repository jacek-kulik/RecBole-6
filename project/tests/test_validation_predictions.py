"""Checkpoint, split identity, and external-ID boundaries using small fake loaders."""

import csv
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from project.recsys.recbole_adapter import _export_splits, _load_validation_checkpoint, validation_predictions


class Values:
    """Minimal array boundary so the standard-library suite needs no torch."""

    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def numpy(self):
        return self.values

    def tolist(self):
        return self.values


class ValidationPredictionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.splits = Path(self.directory.name) / "splits"
        tokens = {"uid": ["[PAD]", "user-b", "user-a"], "iid": ["[PAD]", "a", "b", "c", "d"]}

        def id2token(field, ids):
            if ids and isinstance(ids[0], list):
                return [[tokens[field][item] for item in row] for row in ids]
            return [tokens[field][item] for item in ids]

        self.dataset = SimpleNamespace(uid_field="uid", iid_field="iid", item_num=5, id2token=id2token)

        def loader(items):
            return SimpleNamespace(is_sequential=False, dataset=SimpleNamespace(inter_feat={
                "uid": Values([1, 2]), "iid": Values(items),
                "rating": Values([0.5, 1.0]), "time": Values([10, 20]),
            }))

        self.train, self.valid, self.test = loader([4, 4]), loader([2, 3]), loader([1, 2])
        self.config = {"RATING_FIELD": "rating", "TIME_FIELD": "time", "device": "cpu",
                       "eval_args": {"mode": {"valid": "full", "test": "full"}}}
        self.split_id = _export_splits(self.dataset, (self.train, self.valid, self.test), self.splits, self.config)
        self.model = object()
        self.load_patch = patch("project.recsys.recbole_adapter._load_validation_checkpoint",
                                return_value=(self.config, self.model, self.dataset, self.train, self.valid, self.test))
        self.load = self.load_patch.start()
        self.addCleanup(self.load_patch.stop)

        def topk(batch, model, data, k, device):
            items = {1: [1, 2], 2: [3, 1]}
            return Values([[0.9, 0.8][:k] for _ in batch]), Values([items[user][:k] for user in batch])

        self.topk = Mock(side_effect=topk)
        modules = {"recbole.utils.case_study": SimpleNamespace(full_sort_topk=self.topk)}
        self.modules = patch.dict(sys.modules, modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def predict(self, **kwargs):
        return validation_predictions("selected.pth", self.splits, self.split_id, **kwargs)

    def test_export_preserves_external_ids_ratings_and_timestamps(self):
        with (self.splits / "valid.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows, [
            {"user_id": "user-b", "item_id": "b", "rating": "0.5", "timestamp": "10"},
            {"user_id": "user-a", "item_id": "c", "rating": "1.0", "timestamp": "20"},
        ])

    def test_batches_validation_users_and_translates_both_axes(self):
        config, rankings, relevant = self.predict(k=2, batch_size=1)
        self.assertEqual(config, self.config)
        self.assertEqual(rankings, {"user-b": ["a", "b"], "user-a": ["c", "a"]})
        self.assertEqual(relevant, {"user-b": {"b"}, "user-a": {"c"}})
        self.assertEqual(self.topk.call_count, 2)
        for call in self.topk.call_args_list:
            self.assertIs(call.args[1], self.model)
            self.assertIs(call.args[2], self.valid)
            self.assertEqual(call.kwargs, {"k": 2, "device": "cpu"})
        self.assertEqual([call.args[0] for call in self.topk.call_args_list], [[1], [2]])

    def test_rejects_changed_saved_or_reconstructed_splits_before_scoring(self):
        for name in ("train", "valid", "test"):
            path = self.splits / f"{name}.csv"
            original = path.read_bytes()
            path.write_bytes(original + b"changed\n")
            with self.subTest(partition=name), self.assertRaisesRegex(ValueError, "splits"):
                self.predict(k=2)
            path.write_bytes(original)
        self.valid.dataset.inter_feat["iid"] = Values([1, 3])
        with self.assertRaisesRegex(ValueError, "splits"):
            self.predict(k=2)
        self.topk.assert_not_called()

    def test_rejects_wrong_split_hash(self):
        self.split_id = "different"
        with self.assertRaisesRegex(ValueError, "splits"):
            self.predict(k=2)
        self.topk.assert_not_called()

    def test_rejects_nonfinite_topk_scores(self):
        for score in (float("nan"), float("inf"), float("-inf")):
            self.topk.side_effect = None
            self.topk.return_value = Values([[score, 1.0]]), Values([[1, 2]])
            with self.subTest(score=score), self.assertRaisesRegex(ValueError, "finite"):
                self.predict(k=2, batch_size=1)

    def test_rejects_unsupported_evaluation_and_oversized_cutoff(self):
        self.config["eval_args"]["mode"]["valid"] = "uni100"
        with self.assertRaisesRegex(ValueError, "full-sort"):
            self.predict(k=2)
        self.config["eval_args"]["mode"]["valid"] = "full"
        self.valid.is_sequential = True
        with self.assertRaisesRegex(ValueError, "non-sequential"):
            self.predict(k=2)
        self.valid.is_sequential = False
        with self.assertRaisesRegex(ValueError, "catalogue"):
            self.predict(k=5)
        self.topk.assert_not_called()

    def test_rejects_invalid_limits_before_loading(self):
        for kwargs in ({"k": 0}, {"batch_size": 0}, {"k": True}, {"batch_size": "1"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.predict(**kwargs)
        self.load.assert_not_called()


class CheckpointLoadingTests(unittest.TestCase):
    def test_restores_selected_weights_and_reconstructs_on_cpu(self):
        config = {"device": "cuda", "use_gpu": True, "seed": 42, "reproducibility": True, "model": "BPR"}
        saved = {"config": config, "state_dict": {"weights": "best"}, "other_parameter": "extra"}
        torch = SimpleNamespace(load=Mock(return_value=saved), device=Mock(return_value="cpu"))
        dataset = object()
        train = SimpleNamespace(dataset=dataset)
        valid, test = object(), object()
        create_dataset = Mock(return_value=dataset)
        preparation = Mock(return_value=(train, valid, test))
        model = Mock()
        model.to.return_value = model
        constructor = Mock(return_value=model)
        get_model, seed = Mock(return_value=constructor), Mock()
        modules = {
            "torch": torch,
            "recbole.data": SimpleNamespace(create_dataset=create_dataset, data_preparation=preparation),
            "recbole.utils": SimpleNamespace(get_model=get_model, init_seed=seed),
        }
        with patch.dict(sys.modules, modules):
            result = _load_validation_checkpoint(Path("best.pth"))
        torch.load.assert_called_once_with("best.pth", map_location="cpu", weights_only=False)
        self.assertEqual(config["device"], "cpu")
        self.assertFalse(config["use_gpu"])
        constructor.assert_called_once_with(config, dataset)
        preparation.assert_called_once_with(config, dataset)
        model.load_state_dict.assert_called_once_with(saved["state_dict"])
        model.load_other_parameter.assert_called_once_with("extra")
        self.assertEqual(seed.call_count, 2)
        self.assertEqual(result, (config, model, dataset, train, valid, test))


if __name__ == "__main__":
    unittest.main()