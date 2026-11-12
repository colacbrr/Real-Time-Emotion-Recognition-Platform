#!/usr/bin/env python3
"""Romanian presentation assets from frozen v2 results; no fitting or new metrics."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs/publicare/studiu_v2/results'
TARGET = ROOT / 'docs/publicare/prezentare_proiect/figuri'
NAMES = {'majority': 'Majoritar', 'hog_linear': 'HOG + SVM liniar', 'hog_rbf': 'HOG + SVM RBF',
         'resnet_linear': 'ResNet18 + SVM liniar', 'resnet_rbf': 'ResNet18 + SVM RBF'}
CLASSES = {'angry': 'Furie', 'disgust': 'Dezgust', 'fear': 'Frică', 'happy': 'Bucurie',
           'neutral': 'Neutru', 'sad': 'Tristețe', 'surprise': 'Surpriză'}


def number(value, digits):
    return f'{value:.{digits}f}'.replace('.', ',')


def main():
    for name, sha in json.loads((SOURCE / 'integrity.json').read_text()).items():
        if hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() != sha:
            raise ValueError('Changed numeric evidence: ' + name)
    r = json.loads((SOURCE / 'results.json').read_text())
    TARGET.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'font.family': 'DejaVu Sans',
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.2), layout='constrained')
    for ax, scope, title in zip(axes, ['oof', 'holdout'],
                               ['Validare încrucișată · 2069 imagini', 'Evaluare separată · 509 imagini']):
        e = r['evaluations'][scope]
        for i, model in enumerate(NAMES):
            point = e['models'][model]['macro_f1']
            lo, hi = e['uncertainty']['model_intervals_95'][model]['macro_f1']
            ax.errorbar(point, i, xerr=[[max(0, point-lo)], [max(0, hi-point)]],
                        fmt='o', color='#24576B', capsize=3, markersize=5)
        ax.set_yticks(range(len(NAMES)), list(NAMES.values()))
        ax.invert_yaxis()
        ax.set(xlabel='Macro-F1', title=title, xlim=(0, .46))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: number(v, 1)))
        ax.grid(axis='x', alpha=.15)
    fig.savefig(TARGET / 'macro_f1_ro.png', dpi=300)
    fig.savefig(TARGET / 'macro_f1_ro.svg')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.85), layout='constrained')
    for ax, name in zip(axes, ['hog_rbf', 'resnet_rbf']):
        m = r['evaluations']['holdout']['models'][name]
        cm = np.array(m['confusion_matrix'])
        fraction = cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(fraction, vmin=0, vmax=1, cmap='Blues')
        for i in range(7):
            for j in range(7):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=10,
                        color='white' if fraction[i, j] > .55 else '#17212B')
        labels = [CLASSES[label] for label in m['labels']]
        ax.set_xticks(range(7), labels, rotation=35, ha='right')
        ax.set_yticks(range(7), labels)
        ax.set(title=NAMES[name], xlabel='Clasa prezisă', ylabel='Clasa de referință')
    bar = fig.colorbar(im, ax=axes, shrink=.85, pad=.02)
    bar.set_label('Proporție în clasa de referință')
    bar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: number(v, 1)))
    fig.savefig(TARGET / 'confuzii_rbf_ro.png', dpi=300)
    fig.savefig(TARGET / 'confuzii_rbf_ro.svg')
    plt.close(fig)

    lines = [r'\begin{tabular}{lrrrr}', r'\toprule',
             r'& \multicolumn{2}{c}{Validare încrucișată} & \multicolumn{2}{c}{Evaluare separată} \\',
             r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
             r'Model & Acuratețe (\%) & Macro-F1 & Acuratețe (\%) & Macro-F1 \\', r'\midrule']
    for name in NAMES:
        o, h = [r['evaluations'][s]['models'][name] for s in ['oof', 'holdout']]
        values = [number(o['accuracy']*100, 2), number(o['macro_f1'], 4),
                  number(h['accuracy']*100, 2), number(h['macro_f1'], 4)]
        lines.append(NAMES[name] + ' & ' + ' & '.join(values) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (TARGET / 'scoruri.tex').write_text('\n'.join(lines) + '\n')
    e = r['evaluations']['holdout']['models']
    lines = [r'\begin{tabular}{lrrrrr}', r'\toprule',
             r'Clasă & Imagini & HOG liniar & HOG RBF & ResNet liniar & ResNet RBF \\', r'\midrule']
    for label in CLASSES:
        values = [str(e['hog_rbf']['support_per_class'][label])] + [
            number(100*e[k]['recall_per_class'][label], 1) for k in list(NAMES)[1:]]
        lines.append(CLASSES[label] + ' & ' + ' & '.join(values) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (TARGET / 'recall.tex').write_text('\n'.join(lines) + '\n')
    evidence = {'numeric_source_sha256': hashlib.sha256((SOURCE / 'results.json').read_bytes()).hexdigest(),
                'label_translation': CLASSES, 'metrics_recomputed': False, 'models_fitted': False}
    (TARGET / 'provenienta.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')
    print('Generated Romanian figures and tables from verified frozen results.')


if __name__ == '__main__':
    main()
