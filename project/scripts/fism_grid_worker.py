"""Evaluate an independent grid block and publish completed matching trials."""

import argparse
import json
import logging
from pathlib import Path

import torch
from threadpoolctl import threadpool_limits

from project.scripts import refresh_fism_oof as refresh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--old', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--stop', type=int, required=True)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    worker = root / f'worker-{args.start}-{args.stop}'
    worker.mkdir(exist_ok=True)
    main_identity = json.loads((root / 'tuning/identity.json').read_text())
    grid = refresh.search_space()[args.start:args.stop]
    refresh.search_space = lambda: grid
    write = refresh.atomic_json

    def publish(path, value):
        write(path, value)
        if path.parent.name == 'tuning' and path.name.startswith('trial-'):
            identity = json.loads((worker / 'tuning/identity.json').read_text())
            assert identity == main_identity
            index = args.start + int(path.stem.split('-')[1])
            target = root / 'tuning' / f'trial-{index:02d}.json'
            if not target.exists() and not (root / 'tuning' / f'trial-{index:02d}-training').exists():
                write(target, value)
                print(f'Published original grid trial {index + 1}/48', flush=True)
            elif target.exists():
                existing = json.loads(target.read_text())
                assert existing['parameters'] == value['parameters']
                assert existing['validation'] == value['validation']

    refresh.atomic_json = publish
    torch.set_num_threads(2)
    logging.basicConfig(level=logging.ERROR)
    write(worker / 'worker-provenance.json', {'grid_start_inclusive': args.start,
          'grid_stop_exclusive': args.stop, 'worker_sha256': refresh.file_hash(__file__),
          'independent_process': True, 'selection_performed': False})
    with threadpool_limits(limits=2):
        try:
            refresh.retune(Path(args.old).resolve(), worker)
        except StopIteration:
            # A block need not contain the historical control. Only the main
            # process performs selection after all 48 trials have completed.
            pass
    print('Worker grid block complete', flush=True)


if __name__ == '__main__':
    main()
