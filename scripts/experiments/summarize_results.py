"""Plot the current research run and collect its stage metrics."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def plot_classification(metrics, output):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for axis, view in zip(axes, ('full', 'roi')):
        result = metrics[view]
        matrix = np.array(result['confusion_matrix'])
        labels = result['rows_true']
        axis.imshow(matrix, vmin=0, cmap='Blues')
        axis.set_xticks(range(len(labels) + 1), labels + ['unavailable'], rotation=40, ha='right')
        axis.set_yticks(range(len(labels)), labels)
        for (row, col), count in np.ndenumerate(matrix):
            axis.text(col, row, count, ha='center', va='center',
                      color='white' if count > matrix.max() / 2 else '#183342')
        axis.set(xlabel='Predicted class / status', ylabel='True class',
                 title=f'{view.upper()}   n={result["n"]}   macro F1={result["macro_f1"]:.4f}')
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_reliability(root, stages, output):
    fig, axes = plt.subplots(1, len(stages), figsize=(6 * len(stages), 4.5), squeeze=False,
                             constrained_layout=True)
    for axis, stage in zip(axes[0], stages):
        file = root / stage / 'predictions.jsonl'
        rows = [json.loads(line) for line in file.read_text(encoding='utf-8').splitlines() if line.strip()]
        completed = [row for row in rows if row.get('status') == 'completed']
        axis.scatter([row['mean_variance'] for row in completed],
                     [1 - row['dice'] for row in completed], color='#147D81')
        axis.axhline(.3, color='#AF6E40', linestyle='--', label='Dice = 0.7')
        axis.set(xlabel='Mean TTA variance', ylabel='Segmentation error (1 - Dice)',
                 title=f'{stage}   n={len(completed)}')
        axis.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        axis.legend(fontsize=8)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs/research-v1'))
    args = parser.parse_args()
    root = args.root
    stages = ('segmentation', 'classification', 'reliability', 'reliability-baseline')
    summaries = {stage: read_json(root / stage / 'summary.json') for stage in stages
                 if (root / stage / 'summary.json').is_file()}
    audit = root / 'report-audit-v1.json'
    if audit.is_file():
        summaries['reports'] = read_json(audit)['summary']
    if not summaries:
        parser.error(f'No completed experiment summaries found in {root}')
    figures = root / 'figures'
    figures.mkdir(exist_ok=True)
    if 'classification' in summaries:
        plot_classification(summaries['classification'], figures / 'classification-confusion.png')
    reliability = [stage for stage in ('reliability', 'reliability-baseline') if stage in summaries]
    if reliability:
        plot_reliability(root, reliability, figures / 'reliability-error.png')
    destination = root / 'summary-v1.json'
    destination.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding='utf-8')
    print(destination)


if __name__ == '__main__':
    main()
