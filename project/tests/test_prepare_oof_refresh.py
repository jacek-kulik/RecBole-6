"""Frozen-source preparation must preserve byte identity across Git newline settings."""

import hashlib
import tempfile
import unittest
from pathlib import Path

from project.scripts.prepare_oof_refresh import restore_recorded_bytes


class PrepareOOFRefreshTests(unittest.TestCase):
    def test_archived_crlf_dataset_restored_to_recorded_lf(self):
        expected = b'user_id\titem_id\trating\n1\t2\t5\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ml-100k.inter'
            path.write_bytes(expected.replace(b'\n', b'\r\n'))
            restore_recorded_bytes(path, hashlib.sha256(expected).hexdigest(), 'Dataset')
            self.assertEqual(path.read_bytes(), expected)

    def test_recorded_crlf_and_already_matching_dataset_preserved(self):
        expected = b'user_id\titem_id\trating\r\n1\t2\t5\r\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ml-100k.inter'
            for original in (expected.replace(b'\r\n', b'\n'), expected):
                path.write_bytes(original)
                restore_recorded_bytes(path, hashlib.sha256(expected).hexdigest(), 'Dataset')
                self.assertEqual(path.read_bytes(), expected)

    def test_changed_dataset_contents_are_rejected_without_rewriting(self):
        expected = b'1\t2\t5\n'
        changed = b'1\t2\t4\r\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ml-100k.inter'
            path.write_bytes(changed)
            with self.assertRaisesRegex(ValueError, 'Dataset differs'):
                restore_recorded_bytes(path, hashlib.sha256(expected).hexdigest(), 'Dataset')
            self.assertEqual(path.read_bytes(), changed)


if __name__ == '__main__':
    unittest.main()
