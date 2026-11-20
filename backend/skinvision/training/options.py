"""Small cross-platform CLI shared by the existing train/test entrypoints."""
import argparse
import torch


def select_device(config):
    requested = getattr(config, 'device', 'auto')
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use --device cpu')
    return torch.device(requested)


def configure(defaults):
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--data-path')
    parser.add_argument('--work-dir')
    parser.add_argument('--test-weights')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--profile', action='store_true', help='Optional FLOPs profiling (expensive on CPU)')
    args = parser.parse_args()
    config = defaults()
    for key, value in vars(args).items():
        if value is not None:
            setattr(config, key, value)
    if args.threads < 1 or config.num_workers < 0 or config.batch_size < 1 or config.epochs < 1:
        parser.error('threads/batch-size/epochs must be positive and num-workers nonnegative')
    torch.set_num_threads(args.threads)
    return config
