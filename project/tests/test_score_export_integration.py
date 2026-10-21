"""Real RecBole training, checkpoint selection, masking, and score-file boundary."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from project.recsys import recbole_adapter as adapter
from recbole.trainer import Trainer


class ScoreExportIntegrationTests(unittest.TestCase):
    def test_train_exports_selected_checkpoint_with_real_partition_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "dataset/tiny"
            data.mkdir(parents=True)
            rows = ["user_id:token\titem_id:token\trating:float\ttimestamp:float"]
            for user in range(4):
                for offset in range(4):
                    rows.append(f"user-{user}\tmovie-{(2 * user + offset) % 8}\t1\t{offset}")
            (data / "tiny.inter").write_text("\n".join(rows) + "\n", encoding="utf-8")
            configs = root / "project/configs"
            configs.mkdir(parents=True)
            protocol = {
                "dataset": "tiny", "seed": 42, "reproducibility": True,
                "use_gpu": False, "show_progress": False,
                "load_col": {"inter": ["user_id", "item_id", "rating", "timestamp"]},
                "eval_args": {"split": {"RS": [0.5, 0.25, 0.25]},
                              "group_by": "user", "order": "RO", "mode": "full"},
                "metrics": ["NDCG"], "topk": [2], "valid_metric": "NDCG@2",
                "save_dataset": False, "save_dataloaders": False,
                "worker": 0, "enable_amp": False,
            }
            (configs / "protocol.yaml").write_text(yaml.safe_dump(protocol), encoding="utf-8")
            model_config = configs / "bpr.yaml"
            model_config.write_text(yaml.safe_dump({
                "model": "BPR", "embedding_size": 4, "epochs": 2,
                "eval_step": 1, "train_batch_size": 8,
            }), encoding="utf-8")
            fit = Trainer.fit
            changed = []

            def fit_then_change_live_weights(trainer, *args, **kwargs):
                result = fit(trainer, *args, **kwargs)
                # Distinguish the selected checkpoint from the live model at export.
                with torch.no_grad():
                    for parameter in trainer.model.parameters():
                        parameter.add_(100)
                changed.append(True)
                return result

            output = root / "run"
            with patch.object(adapter, "REPO_ROOT", root), patch.object(
                Trainer, "fit", fit_then_change_live_weights
            ), patch("recbole.utils.init_logger"), patch(
                "recbole.trainer.trainer.get_tensorboard",
                side_effect=lambda logger: SummaryWriter(log_dir=str(root / "tensorboard")),
            ):
                adapter.train(model_config, output)
                manifest = json.loads((output / "manifest.json").read_text())
                _, restored, dataset, _, _, _ = adapter._load_checkpoint(output / manifest["checkpoint"])

            self.assertEqual(changed, [True])
            self.assertEqual(manifest["status"], "complete")
            self.assertFalse(manifest["test_evaluated"])
            histories = {}
            for partition in ("train", "valid", "test"):
                with (output / "splits" / f"{partition}.csv").open(newline="") as stream:
                    histories[partition] = {}
                    for row in csv.DictReader(stream):
                        histories[partition].setdefault(row["user_id"], set()).add(row["item_id"])
            catalogue = set(dataset.id2token(dataset.iid_field, list(range(1, dataset.item_num))))
            for partition in ("valid", "test"):
                # Exercise the actual gzip writer in train and its public reader.
                table = adapter.load_scores(output / manifest["score_files"][partition])
                self.assertEqual(table.split_id, manifest["split_id"])
                self.assertEqual(table.partition, partition)
                self.assertEqual(set(table.scores), set(histories[partition]))
                for user, scores in table.scores.items():
                    seen = histories["train"][user].copy()
                    if partition == "test":
                        seen.update(histories["valid"][user])
                    self.assertEqual(set(scores), catalogue - seen)
                    uid = dataset.token2id(dataset.uid_field, user)
                    for item, score in scores.items():
                        iid = dataset.token2id(dataset.iid_field, item)
                        with torch.no_grad():
                            expected = (restored.user_embedding.weight[uid]
                                        * restored.item_embedding.weight[iid]).sum().item()
                        self.assertAlmostEqual(score, expected, places=6)


if __name__ == "__main__":
    unittest.main()
