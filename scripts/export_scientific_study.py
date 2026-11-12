#!/usr/bin/env python3
"""Render scientific v2 saved numeric evidence; system Python + matplotlib, no fitting."""
import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

NAMES = {'majority': 'Majority', 'hog_linear': 'HOG + linear SVM', 'hog_rbf': 'HOG + RBF SVM',
         'resnet_linear': 'ResNet18 + linear SVM', 'resnet_rbf': 'ResNet18 + RBF SVM'}


def read(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def csv_out(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def export(a):
    target = a.plan / 'results'
    target.mkdir(exist_ok=False)
    report = json.loads((a.run / 'results.json').read_text())
    for filename in ['results.json', 'oof_predictions.csv', 'holdout_predictions.csv', 'tuning.json']:
        shutil.copyfile(a.run / filename, target / filename)
    all_rows, recall_rows, pairs = [], [], []
    for scope, evaluation in report['evaluations'].items():
        for name, m in evaluation['models'].items():
            ci = evaluation['uncertainty']['model_intervals_95'][name]
            all_rows.append({'scope': scope, 'model': name, 'n': m['n'], 'accuracy': m['accuracy'],
                             'accuracy_lo': ci['accuracy'][0], 'accuracy_hi': ci['accuracy'][1], 'macro_f1': m['macro_f1'],
                             'macro_f1_lo': ci['macro_f1'][0], 'macro_f1_hi': ci['macro_f1'][1]})
            for label, value in m['recall_per_class'].items():
                recall_rows.append({'scope': scope, 'model': name, 'class': label, 'support': m['support_per_class'][label], 'recall': value})
            csv_out(target / f'{scope}_{name}_confusion.csv', [dict(zip(['true_label'] + m['labels'], [label] + row))
                                                            for label, row in zip(m['labels'], m['confusion_matrix'])])
        for pair in evaluation['uncertainty']['paired_differences']:
            for metric in ['accuracy', 'macro_f1']:
                point = evaluation['models'][pair['a']][metric] - evaluation['models'][pair['b']][metric]
                pairs.append({'scope': scope, 'a': pair['a'], 'b': pair['b'], 'metric': metric, 'difference': point,
                              'lo_95': pair['interval_95'][metric][0], 'hi_95': pair['interval_95'][metric][1],
                              'lo_four_pairs': pair['interval_familywise_95_four_pairs'][metric][0],
                              'hi_four_pairs': pair['interval_familywise_95_four_pairs'][metric][1]})
    csv_out(target / 'scores.csv', all_rows)
    csv_out(target / 'recall_per_class.csv', recall_rows)
    csv_out(target / 'paired_differences.csv', pairs)
    membership = defaultdict(set)
    local_split = {}
    for row in read(a.matches):
        membership[row['id']].add(row['reference_split'])
        local_split[row['id']] = row['local_split']
    counts = Counter((local_split[k], '+'.join(sorted(v))) for k, v in membership.items())
    csv_out(target / 'source_membership.csv', [{'academic_split': k[0], 'reference_splits': k[1], 'n': v} for k, v in sorted(counts.items())])
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), constrained_layout=True)
    models = list(NAMES)
    for ax, scope, title in zip(axes, ['oof', 'holdout'], ['Grouped out-of-fold (n=2069)', 'Source-test subset (n=509), retrospective']):
        e = report['evaluations'][scope]
        for i, model in enumerate(models):
            value = e['models'][model]['macro_f1']
            lo, hi = e['uncertainty']['model_intervals_95'][model]['macro_f1']
            ax.errorbar(value, i, xerr=[[max(0, value-lo)], [max(0, hi-value)]], fmt='o', color='#126782', capsize=3)
        ax.set_yticks(range(5), [NAMES[k] for k in models])
        ax.invert_yaxis()
        ax.set(xlabel='Macro-F1; conditional 95% percentile intervals', title=title, xlim=(0, .55))
        ax.grid(axis='x', alpha=.2)
    fig.savefig(target / 'macro_f1.png', dpi=240)
    fig.savefig(target / 'macro_f1.svg')
    plt.close(fig)
    e = report['evaluations']['holdout']
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for ax, model in zip(axes.flat, models[1:]):
        m = e['models'][model]
        raw = np.array(m['confusion_matrix'])
        normalized = raw / raw.sum(axis=1, keepdims=True)
        ax.imshow(normalized, vmin=0, vmax=1, cmap='Blues')
        for i in range(7):
            for j in range(7):
                ax.text(j, i, str(raw[i, j]), ha='center', va='center', fontsize=8,
                        color='white' if normalized[i, j] > .55 else 'black')
        ax.set_xticks(range(7), m['labels'], rotation=45, ha='right')
        ax.set_yticks(range(7), m['labels'])
        ax.set(title=NAMES[model], xlabel='Predicted label', ylabel='True label')
    fig.savefig(target / 'confusion_holdout.png', dpi=240)
    plt.close(fig)
    # LaTeX fragments contain numbers only from saved reports.
    lines = [r'\begin{tabular}{llrr}', r'\toprule', r'Evaluation & Pipeline & Accuracy & Macro-F1 [95\% interval] \\', r'\midrule']
    for r in all_rows:
        scope = 'OOF' if r['scope'] == 'oof' else 'Source-test'
        lines.append(f"{scope} & {NAMES[r['model']]} & {100*r['accuracy']:.2f} & {100*r['macro_f1']:.2f} [{100*r['macro_f1_lo']:.2f}, {100*r['macro_f1_hi']:.2f}] " + r'\\')
    lines.extend([r'\bottomrule', r'\end{tabular}'])
    (target / 'scores.tex').write_text('\n'.join(lines) + '\n')
    lines = [r'\begin{tabular}{lrrrrr}', r'\toprule', r'Class (support) & Majority & HOG lin. & HOG RBF & RN lin. & RN RBF \\', r'\midrule']
    for label in e['models']['majority']['labels']:
        support = e['models']['majority']['support_per_class'][label]
        values = ' & '.join(f"{100*e['models'][k]['recall_per_class'][label]:.1f}" for k in models)
        lines.append(f'{label} ({support}) & {values} ' + r'\\')
    lines.extend([r'\bottomrule', r'\end{tabular}'])
    (target / 'recall.tex').write_text('\n'.join(lines) + '\n')
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.iterdir() if p.is_file()}
    (target / 'integrity.json').write_text(json.dumps(files, indent=2) + '\n')
    print(json.dumps({'output': str(target), 'files': len(files), 'run_seconds': report['elapsed_seconds']}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['plan', 'run', 'matches']:
        p.add_argument('--' + key, type=Path, required=True)
    export(p.parse_args())
