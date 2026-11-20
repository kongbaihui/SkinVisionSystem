"""Exercise the real network/optimizer on synthetic data, never a quality benchmark.

Run: python -m scripts.tools.smoke_training --outdir outputs/smoke
"""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', default='outputs/smoke')
    args = parser.parse_args()
    root = Path(args.outdir).resolve()
    # Refuse reuse so an old checkpoint cannot masquerade as this run.
    root.mkdir(parents=True, exist_ok=False)
    data = root / 'synthetic-data'
    data.mkdir()
    rng = np.random.default_rng(42)
    for split, count in [('train', 2), ('val', 1), ('test', 1)]:
        np.save(data / f'data_{split}.npy', rng.integers(0, 256, (count, 32, 32, 3), dtype=np.uint8))
        mask = np.zeros((count, 32, 32), np.uint8)
        mask[:, 8:24, 8:24] = 255
        np.save(data / f'mask_{split}.npy', mask)
    results = {'purpose': 'engineering smoke only; synthetic data, random initialization',
               'platform': platform.platform(), 'python': sys.version, 'torch': torch.__version__,
               'device': 'cpu', 'seed': 42, 'image_size': [32, 32], 'runs': []}
    for stage in ['train', 'test']:
        command = [sys.executable, '-m', f'backend.skinvision.training.{stage}',
                   '--device', 'cpu', '--data-path', str(data), '--num-workers', '0',
                   '--work-dir', str(root / stage), '--threads', '2']
        if stage == 'train':
            command += ['--epochs', '1', '--batch-size', '2']
        else:
            command += ['--test-weights', str(root / 'train/checkpoints/latest.pth')]
        start = time.perf_counter()
        run = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace',
                             env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
        (root / f'{stage}.log').write_text(run.stdout + run.stderr, encoding='utf-8')
        results['runs'].append({'stage': stage, 'command': command, 'exit_code': run.returncode,
                                'elapsed_seconds': time.perf_counter() - start})
        (root / 'result.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
        print(stage, 'exit:', run.returncode, flush=True)
        if run.returncode:
            print((run.stdout + run.stderr).encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8'))
            raise SystemExit(run.returncode)


if __name__ == '__main__':
    main()
