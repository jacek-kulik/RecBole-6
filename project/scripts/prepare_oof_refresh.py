"""Reconstruct the frozen source checkout used for the final Task 1.3 rerun."""

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path


def matching_bytes(blob, expected):
    """Accept Git LF or Windows checkout CRLF only when the recorded hash matches."""
    for candidate in (blob, blob.replace(b'\r\n', b'\n'), blob.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')):
        if hashlib.sha256(candidate).hexdigest() == expected:
            return candidate
    return None


def restore_recorded_bytes(path, expected, label):
    """Restore only a newline-equivalent file to its exact recorded byte identity."""
    path = Path(path)
    blob = matching_bytes(path.read_bytes(), expected)
    if blob is None:
        raise ValueError(f'{label} differs from the recorded experiment')
    path.write_bytes(blob)


def prepare(output):
    repo = Path(__file__).resolve().parents[2]
    records = repo / 'hyperparameters/results/oof-ridge-final-60-20-20'
    provenance = json.loads((records / 'source-provenance.json').read_text())

    def git(*args):
        return subprocess.check_output(['git', '-c', f'safe.directory={repo.as_posix()}', *args], cwd=repo)

    # Refuse an existing directory to preserve experiment evidence.
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(git('archive', provenance['base_commit']))) as archive:
        archive.extractall(output, filter='data')
    for raw_path, expected in provenance['sha256'].items():
        path = raw_path.replace('\\', '/')
        target = output / path
        if hashlib.sha256(target.read_bytes()).hexdigest() == expected:
            continue
        refs = ([provenance['updated_fism_commit']] if path.endswith('/fism.py')
                else git('log', '--all', '--format=%H', '--', path).decode().splitlines())
        for ref in refs:
            blob = matching_bytes(git('show', f'{ref}:{path}'), expected)
            if blob is not None:
                target.write_bytes(blob)
                break
        else:
            raise ValueError(f'Cannot reconstruct recorded source: {path}; fetch full Git history')
    identity = json.loads((records / 'fism-tuning/identity.json').read_text())
    restore_recorded_bytes(output / 'dataset/ml-100k/ml-100k.inter',
                           identity['dataset_sha256'], 'Dataset')
    protocol = matching_bytes((records / 'frozen-protocol.yaml').read_bytes(), identity['protocol_sha256'])
    if protocol is None:
        raise ValueError('Protocol differs from the recorded experiment')
    (output / 'project/configs/protocol.yaml').write_bytes(protocol)
    for filename in ('refresh_fism_oof.py', 'fism_grid_worker.py', 'fism_oof_fold_worker.py'):
        blob = (repo / 'project/scripts' / filename).read_bytes()
        if filename == 'refresh_fism_oof.py':
            blob = matching_bytes(blob, identity['runner_sha256'])
            if blob is None:
                raise ValueError('Refresh runner differs from the recorded experiment')
        (output / 'project/scripts' / filename).write_bytes(blob)
    shutil.copyfile(records / 'source-provenance.json', output / 'source-provenance.json')
    shutil.copyfile(repo / 'project/tests/test_fism.py', output / 'project/tests/test_fism.py')
    assert hashlib.sha256((output / 'project/configs/protocol.yaml').read_bytes()).hexdigest() == identity['protocol_sha256']
    assert hashlib.sha256((output / 'dataset/ml-100k/ml-100k.inter').read_bytes()).hexdigest() == identity['dataset_sha256']
    print(output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    prepare(parser.parse_args().output)
