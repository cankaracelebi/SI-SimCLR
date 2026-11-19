#!/usr/bin/env python3
"""
Visualization of leave-one-batch-out classification results (output of batch_class.py).
Script version of embeddings_analysis/Supp_2_left_batch_out_visualization.ipynb.

Run from the repository root:
    python experiments/resnet50/batch_classification/batch_viz.py \
        --results results/analysis/batch_classification/results_lobo.csv

Writes next to the results file (or to --out-dir):
    auc_lobo.csv            one AUC per feature set / task / held-out batch
    auc_lobo_boxplot.pdf    AUC distribution per feature set, one panel per task
    auc_lobo_per_batch.pdf  AUC per held-out batch, one panel per task (Supp. Fig. 2A)
    auc_lobo_per_batch_SI_SimCLR.pdf  same, SI-SimCLR only (Supp. Fig. 2B)
"""
import os, sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path[:0] = [REPO_ROOT, os.path.join(REPO_ROOT, 'SI-SimCLR')]

import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
plt.rcParams['pdf.fonttype'] = 42
from utils.mypalette import day_palette, pairing_strategy_palette

# Plot order of the feature sets; only the ones present in the results are drawn.
# Paper order, with the two added natural-image baselines placed after ResNet18.
ORDER = ['hand-crafted', 'Pretrained\nResNet18', 'Pretrained\nResNet50', 'Pretrained\nViT-S',
         'Standard SimCLR', 'uniDINO', 'CellPaint SSL', 'SI SimCLR']
PALETTE = dict(pairing_strategy_palette)
PALETTE['Pretrained\nResNet50'] = '#8e3a80'   # not in the paper palette
PALETTE['Pretrained\nViT-S'] = '#b5482f'      # not in the paper palette

TASKS = ['ALS', 'TIC low', 'TIC high']
POS_LABEL = {'ALS': 'als', 'TIC high': 'A1 high', 'TIC low': 'A1 low'}


def calculate_aucs(data, group_cols=('exp_code', 'classifier', 'test_batch')):
    """One AUC per group, pooling the predictions of all seeds in that group."""
    group_cols = list(group_cols)
    auc_results = []
    for keys, group in data.groupby(group_cols):
        if group['true'].nunique() < 2:   # AUC needs both classes
            continue
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        task = row.get('classifier', group['classifier'].iloc[0])
        row['auc'] = roc_auc_score((group['true'] == POS_LABEL[task]).astype(int), group['pred_prob'])
        auc_results.append(row)
    return pd.DataFrame(auc_results)


def plot_boxplot(auc_df, order, out_path):
    """Same settings as the paper notebook (catplot boxes + one dot per held-out batch)."""
    g = sns.catplot(
        data=auc_df, x='exp_code', y='auc', col='classifier', col_order=TASKS,
        kind='box', hue='exp_code', palette=PALETTE, dodge=False,
        height=6, aspect=0.8, fliersize=0, hue_order=order, order=order, sharey=True,
    )
    g.map_dataframe(
        sns.stripplot, x='exp_code', y='auc', hue='test_batch', palette=day_palette,
        dodge=False, order=order, hue_order=None, alpha=1, size=8,
        edgecolor='black', linewidth=0.5, marker='o',
    )
    g.set_titles("{col_name}")
    g.set_axis_labels("Embedding Experiment", "AUC")
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
    plt.tight_layout()
    g.fig.savefig(out_path, dpi=300)
    g.fig.savefig(os.path.splitext(out_path)[0] + '.png', dpi=150)
    plt.close(g.fig)


def plot_per_batch(auc_df, order, out_path):
    """Same settings as the paper notebook (bars per held-out batch, no legend)."""
    fig, ax = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for i, cl in enumerate(TASKS):
        sns.barplot(x='test_batch', y='auc', data=auc_df[auc_df['classifier'] == cl],
                    palette=PALETTE, ax=ax[i], hue='exp_code', hue_order=order)
        ax[i].set_ylabel('AUC')
        ax[i].axhline(0.5, color='black', linestyle='--')
        ax[i].set_title(f'{cl}', y=1.05)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        if ax[i].legend_ is not None:
            ax[i].legend_.remove()
    title = plt.suptitle('AUCs for leave-batch-out-experiments')
    plt.tight_layout()
    extra = [title]
    if len(order) > 1:   # one shared legend below the panels (not in the paper notebook)
        handles, labels = ax[0].get_legend_handles_labels()   # the drawn bars, so colours match exactly
        labels = [e.replace('\n', ' ') for e in labels]
        extra += [fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.0),
                            ncol=len(order), frameon=False, fontsize=13,
                            handlelength=1.4, columnspacing=1.8)]
    fig.savefig(out_path, dpi=300, bbox_inches='tight', bbox_extra_artists=extra)
    fig.savefig(os.path.splitext(out_path)[0] + '.png', dpi=150, bbox_inches='tight', bbox_extra_artists=extra)
    plt.close(fig)


def main(args):
    # Several results files (e.g. feature sets classified in separate runs) are concatenated
    df = pd.concat([pd.read_csv(r) for r in args.results], ignore_index=True)
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.results[0]))
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loaded {len(df)} predictions.")
    print(f"Feature sets: {list(df['exp_code'].unique())}")
    print(df.groupby(['classifier', 'test_batch']).size().unstack(fill_value=0))
    n_nan = int(df.isna().any(axis=1).sum())
    if n_nan:
        print(f"WARNING: {n_nan} rows contain NaN values "
              f"(feature sets: {list(df[df.isna().any(axis=1)]['exp_code'].unique())})")

    auc_df = calculate_aucs(df)
    auc_df.to_csv(os.path.join(out_dir, 'auc_lobo.csv'), index=False)

    unknown = [e for e in auc_df['exp_code'].unique() if e not in ORDER]
    order = [e for e in ORDER if e in auc_df['exp_code'].unique()] + unknown
    for e in unknown:   # feature sets without a paper colour
        PALETTE.setdefault(e, '#888888')

    summary = auc_df.pivot_table(index='exp_code', columns='classifier', values='auc', aggfunc='mean')
    print('\nMean AUC over held-out batches:')
    print(summary.reindex(order)[[t for t in TASKS if t in summary.columns]].round(3))

    plot_boxplot(auc_df, order, os.path.join(out_dir, 'auc_lobo_boxplot.pdf'))
    plot_per_batch(auc_df, order, os.path.join(out_dir, 'auc_lobo_per_batch.pdf'))
    # Supplementary Figure 2B: same plot for the SI-SimCLR model alone
    if 'SI SimCLR' in order:
        plot_per_batch(auc_df[auc_df['exp_code'] == 'SI SimCLR'], ['SI SimCLR'],
                       os.path.join(out_dir, 'auc_lobo_per_batch_SI_SimCLR.pdf'))
    print(f'\nSaved AUC table and figures to {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot leave-one-batch-out classification results')
    parser.add_argument('--results', type=str, nargs='+',
                        default=[os.path.join(REPO_ROOT, 'results', 'analysis', 'batch_classification', 'results_lobo.csv')])
    parser.add_argument('--out-dir', type=str, default=None, help='Defaults to the folder of the first --results file')
    main(parser.parse_args())
