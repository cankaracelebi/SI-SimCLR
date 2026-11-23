"""
Fig. 2A: UMAP of the embeddings of each method, coloured by cell_line / mutation / condition / experiment.
Adapted from embeddings_analysis/Fig_2A_UMAP.ipynb, with Pretrained ResNet50 added.

Run from anywhere (paths are built from the location of this file):
    /opt/miniconda3/envs/si-simclr/bin/python experiments/resnet50/fig2a_umap.py
Output: results/all_UMAP_zscore.pdf (+ .png)
"""
import os
import sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(REPO_ROOT)

import numpy as np
import pandas as pd
import pickle
import matplotlib as mpl
mpl.use('Agg')  # script: save the figure, no window
mpl.rcParams['legend.markerscale'] = 1.5
mpl.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
import warnings
from umap import UMAP
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from utils.mypalette import mutation_palette, condition_palette, day_palette, cell_line_palette

# sklearnex (Intel accelerator) is not available on macOS -> removed
# from sklearnex import patch_sklearn
# patch_sklearn()

warnings.filterwarnings('ignore')

res_dir = os.path.join(REPO_ROOT, 'results')
emb_dir = os.path.join(REPO_ROOT, 'embeddings')

days = ['E1', 'E2', 'E3', 'E4']

# Reference annotations: 19 326 crops. Every embedding must have one row per crop, in THIS order.
global_annotations = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'crop16_EF_2405.csv'))
global_annotations['file_name'] = global_annotations['img_code'].apply(lambda x: os.path.basename(x).split('.')[0])

pca = PCA(n_components=20, random_state=0)
low_variance_threshold = 1e-5


##################################################
# Helpers
##################################################

def to_file_name(x):
    """'/some/path/E1-P1-RW2-CW2-F1-CR1_16.tif' -> 'E1-P1-RW2-CW2-F1-CR1_16'"""
    return os.path.basename(str(x)).split('.')[0]


def align_on_annotations(df, name_col):
    """
    The files shared by the authors are not in the order of global_annotations and some contain
    more crops (~21 900 before quality control vs 19 326 after).
    -> keep only the crops of global_annotations, in the same order, using the image name.
    """
    df = df.copy()
    df['file_name'] = df[name_col].apply(to_file_name)
    df = df.drop_duplicates('file_name').set_index('file_name')
    missing = set(global_annotations['file_name']) - set(df.index)
    assert not missing, f'{len(missing)} crops of crop16_EF_2405.csv are missing in this file'
    df = df.loc[global_annotations['file_name']].reset_index()
    assert df['file_name'].equals(global_annotations['file_name'])
    return df


def emb_columns(df):
    return [c for c in df.columns if c.startswith('emb')]


def run_umap(features):
    return UMAP(n_components=2, random_state=0, n_neighbors=15).fit_transform(features)


##################################################
# Embeddings -> z-score -> UMAP
##################################################

UMAP_features = {}
PCA_features = {}

# --- Standard SimCLR and Augmented / SI-SimCLR: CSVs with img_code, shared by the authors ---
for csv_name, method in [('SimCLR.csv', 'Standard\nSimCLR'), ('SI_SimCLR.csv', 'Augmented\nSimCLR')]:
    simclr_df = align_on_annotations(pd.read_csv(os.path.join(emb_dir, csv_name)), 'img_code')
    scaled_emb = StandardScaler().fit_transform(simclr_df[emb_columns(simclr_df)].values)
    UMAP_features[method] = run_umap(scaled_emb)
    PCA_features[method] = pca.fit_transform(scaled_emb)
    print(method.replace('\n', ' '), 'done')

# --- Hand-crafted features: img_code + numeric features only ---
handcrafted_df = align_on_annotations(pd.read_csv(os.path.join(emb_dir, 'handcrafted.csv')), 'img_code')
handcrafted_feat = handcrafted_df.drop(['img_code', 'file_name'], axis=1)
handcrafted_feat = handcrafted_feat.loc[:, handcrafted_feat.var() != 0].values
handcrafted_feat = StandardScaler().fit_transform(handcrafted_feat)
UMAP_features['Hand-crafted'] = run_umap(handcrafted_feat)
print('Hand-crafted done')

# --- Pretrained ResNet18: vectors already in columns emb0..emb511 (no fix_embedding_format needed) ---
pretrain_resnet_df = align_on_annotations(pd.read_csv(os.path.join(emb_dir, 'pretrained_resnet_embeddings.csv')), 'file_name')
pretrained_emb = pretrain_resnet_df[emb_columns(pretrain_resnet_df)].values
scaled_pretrained = StandardScaler().fit_transform(pretrained_emb)
UMAP_features['Pretrained\nResNet18'] = run_umap(scaled_pretrained)
print('ResNet18 done')

# --- Pretrained ResNet50 (ours, experiments/resnet50/extract_resnet50.py) ---
# Extracted with shuffle=False from crop16_EF_2405.csv -> already in the order of global_annotations.
with open(os.path.join(emb_dir, 'resnet50_imagenet', 'latent_embeddings_train.pickle'), 'rb') as f:
    resnet50_emb = pickle.load(f)
assert len(resnet50_emb) == len(global_annotations)
scaled_resnet50 = StandardScaler().fit_transform(resnet50_emb)
UMAP_features['Pretrained\nResNet50'] = run_umap(scaled_resnet50)
print('ResNet50 done')

# --- uniDINO: image path in column 'path'; low-variance dimensions removed ---
uni_emb = align_on_annotations(pd.read_csv(os.path.join(emb_dir, 'uniDINO.csv')), 'path')
uni_emb_feat = uni_emb[emb_columns(uni_emb)]
high_variance_mask = (uni_emb_feat.var(axis=0) > low_variance_threshold).values
scaled_uni_highvar = StandardScaler().fit_transform(uni_emb_feat.values[:, high_variance_mask])
UMAP_features['uniDINO'] = run_umap(scaled_uni_highvar)
print('uniDINO done')

# --- CellPaint SSL: image path in column 'img_code'; low-variance dimensions removed ---
cellpaint_df = align_on_annotations(pd.read_csv(os.path.join(emb_dir, 'CellPaint-SSL.csv')), 'img_code')
cellpaint_emb = cellpaint_df[emb_columns(cellpaint_df)]
high_variance_mask = (cellpaint_emb.var(axis=0) > low_variance_threshold).values
scaled_cellpaint_highvar = StandardScaler().fit_transform(cellpaint_emb.values[:, high_variance_mask])
UMAP_features['CellPaint SSL'] = run_umap(scaled_cellpaint_highvar)
print('CellPaint SSL done')


####################################################
# Figure, laid out like Fig. 2A of the paper:
# one column per method (coloured header), one row per label, legends at the bottom
####################################################
from matplotlib.lines import Line2D

# (key in UMAP_features, header shown on the figure, header colour, header text colour)
# Order and colours follow the paper; our Pretrained ResNet50 is added next to ResNet18.
METHODS = [
    ('Hand-crafted',         'Classic\nimage analysis', '#A9BCD6', 'black'),
    ('uniDINO',              'ViT-S\nuniDINO',          '#7CC88C', 'black'),
    ('CellPaint SSL',        'ViT-S\nCellPaint SSL',    '#2F9A47', 'black'),
    ('Pretrained\nResNet18', 'Pre-trained\nResNet18',   '#D4D4D4', 'black'),
    ('Pretrained\nResNet50', 'Pre-trained\nResNet50',   '#A6A6A6', 'black'),
    ('Standard\nSimCLR',     'SimCLR\nResNet18',        '#000000', 'white'),
    ('Augmented\nSimCLR',    'SI-SimCLR\nResNet18',     '#B39C3D', 'white'),
]
METHODS = [m for m in METHODS if m[0] in UMAP_features]  # only the methods that were computed

# (column of the annotation CSV, row title, legend title, {value in the CSV: name in the legend})
# Add ('cell_line', 'Cell line', 'Cell line', {...}) here if you also want the cell lines.
ROWS = [
    ('experiment', 'Batch',        'Batch',   {'E1': 'E1', 'E2': 'E2', 'E3': 'E3', 'E4': 'E4'}),
    ('condition',  'Inflammation', 'TIC',     {'untreated': 'Basal', 'A1 low': 'Low', 'A1 high': 'High'}),
    ('mutation',   'ALS disease',  'Disease', {'CTRL': 'CTRL', 'VCP': 'ALS'}),
]
palettes = {'cell_line': cell_line_palette, 'mutation': mutation_palette,
            'condition': condition_palette, 'experiment': day_palette}

n_cols, n_rows = len(METHODS), len(ROWS)
fig = plt.figure(figsize=(2.4 * n_cols, 2.4 * n_rows + 1.6))
gs = fig.add_gridspec(n_rows, n_cols, left=0.05, right=0.99, top=0.84, bottom=0.2, wspace=0.06, hspace=0.06)

for j, (key, header, header_colour, text_colour) in enumerate(METHODS):
    xy = UMAP_features[key]
    for i, (column, row_title, _, _) in enumerate(ROWS):
        ax = fig.add_subplot(gs[i, j])
        ax.scatter(xy[:, 0], xy[:, 1],
                   c=global_annotations[column].map(palettes[column]).fillna('#808080'),  # e.g. CTRL3 -> grey
                   s=1.5, alpha=0.4, linewidths=0, rasterized=True)
        ax.axis('off')
        if i == 0:   # coloured method header above the first row
            ax.text(0.5, 1.06, header, transform=ax.transAxes, ha='center', va='bottom',
                    fontsize=11, color=text_colour,
                    bbox=dict(boxstyle='round,pad=0.35', fc=header_colour, ec='none'))
        if j == 0:   # row title on the left
            ax.text(-0.06, 0.5, row_title, transform=ax.transAxes, ha='right', va='center',
                    rotation=90, fontsize=12)

fig.text(0.01, 0.97, 'A', fontsize=18, fontweight='bold', va='top')

# UMAP axes arrows (bottom-left); UMAP axes have no units, only the directions matter
arrow_ax = fig.add_axes([0.04, 0.115, 0.04, 0.06])
arrow_ax.axis('off')
arrow_ax.annotate('', xy=(0, 1), xytext=(0, 0), xycoords='axes fraction',
                  arrowprops=dict(arrowstyle='-|>', lw=2, color='black'))
arrow_ax.annotate('', xy=(1, 0), xytext=(0, 0), xycoords='axes fraction',
                  arrowprops=dict(arrowstyle='-|>', lw=2, color='black'))

# One framed legend per row, side by side at the bottom
legend_x = np.linspace(0.25, 0.8, n_rows)
for x, (column, _, legend_title, names) in zip(legend_x, ROWS):
    handles = [Line2D([0], [0], marker='s', linestyle='', markersize=9,
                      markerfacecolor=palettes[column].get(value, '#808080'), markeredgewidth=0, label=name)
               for value, name in names.items()]
    leg = fig.legend(handles=handles, title=legend_title, loc='upper center', bbox_to_anchor=(x, 0.16),
                     ncol=len(handles), frameon=True, fontsize=10, title_fontsize=11,
                     handletextpad=0.3, columnspacing=0.8)
    leg.get_frame().set_edgecolor('#BBBBBB')
    leg.get_frame().set_linewidth(1.5)

out_pdf = os.path.join(res_dir, 'all_UMAP_zscore.pdf')
fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
fig.savefig(out_pdf.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
print('Saved', out_pdf)
