"""Checks for reproducible, inspectable run artifacts."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from project.recsys.artifacts import fingerprint, new_run, provenance, write_json


class ArtifactTests(unittest.TestCase):
    def test_fingerprint_is_independent_of_mapping_order(self):
        payload = {"b": ["x", "y"], "a": {"v": 1}}
        self.assertEqual(fingerprint(payload), fingerprint({"a": {"v": 1}, "b": ["x", "y"]}))
        self.assertEqual(fingerprint(payload), hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest())
        self.assertNotEqual(fingerprint(payload), fingerprint({"b": ["y", "x"], "a": {"v": 1}}))

    def test_write_json_has_sorted_keys_and_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            payload = {"z": [0.5, None], "a": {"u": "item"}}
            write_json(str(path), payload)
            self.assertEqual(json.loads(path.read_text()), payload)
            self.assertTrue(path.read_text().endswith("\n"))
            self.assertLess(path.read_text().index('"a"'), path.read_text().index('"z"'))

    def test_nonfinite_json_is_rejected_without_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text("existing evidence")
            for value in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    write_json(path, {"metric": value})
                self.assertEqual(path.read_text(), "existing evidence")

    def test_new_run_creates_parents_and_protects_existing_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "run"
            self.assertEqual(new_run(str(path)), path.resolve())
            evidence = path / "evidence.json"
            evidence.write_text("saved")
            with self.assertRaises(FileExistsError):
                new_run(path)
            self.assertEqual(evidence.read_text(), "saved")
            with self.assertRaises(FileExistsError):
                new_run(evidence)

    def test_provenance_records_clean_dirty_and_unavailable_git(self):
        for status, returncode, dirty in (("", 0, False), (" M tracked.py\n", 0, True), ("", 1, None)):
            with self.subTest(dirty=dirty):
                responses = [
                    subprocess.CompletedProcess([], returncode, status, ""),
                    subprocess.CompletedProcess([], returncode, "abc123\n" if not returncode else "", ""),
                ]
                with patch("project.recsys.artifacts.subprocess.run", side_effect=responses):
                    result = provenance()
                self.assertEqual(result["git_dirty"], dirty)
                self.assertEqual(result["git_commit"], "abc123" if not returncode else None)
                self.assertEqual(result["python"], sys.version)
                self.assertEqual(result["command"], sys.argv)


if __name__ == "__main__":
    unittest.main()
