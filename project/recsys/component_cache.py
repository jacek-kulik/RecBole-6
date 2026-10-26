"""Small verified, atomic stage cache for expensive OOF component training."""

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temp, path)


def save_fold(path, indices, values):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, indices=indices, values=values)
    os.replace(temp, path)
    atomic_json(path.with_suffix('.complete.json'), {'sha256': file_hash(path)})


def load_fold(path, indices):
    path = Path(path)
    marker = path.with_suffix('.complete.json')
    if not marker.exists():
        return None
    if not path.exists() or file_hash(path) != json.loads(marker.read_text())['sha256']:
        raise ValueError(f'Corrupt completed cache: {path}')
    with np.load(path, allow_pickle=False) as saved:
        values = saved['values'].copy()
        if (not np.array_equal(saved['indices'], indices) or values.shape != indices.shape
                or not np.isfinite(values).all()):
            raise ValueError(f'Invalid fold cache: {path}')
    return values


@contextmanager
def run_lock(output):
    path = Path(output) / 'run.lock'
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise ValueError(f'Run is locked: {path}. If interrupted, verify its PID stopped before removing the lock.')
    try:
        os.write(descriptor, str(os.getpid()).encode())
        os.close(descriptor)
        yield
    finally:
        path.unlink()
