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

# --- Standard SimCLR (results/Standard SimCLR) ---
# Skipped: the pickle has 21 944 rows and no image names; its row order is given by
# crop16_EF_2405_adj_single.csv, which was not shared. Without it every point would get
# the colour of another crop.

# --- Augmented SimCLR / SI-SimCLR (results/SI-SimCLR) ---
# The pickle has no image names: its 21 872 rows follow the row order of uniDINO.csv
# (same annotation file, crop16_EF_2405, used for training).
uni_order = pd.read_csv(os.path.join(emb_dir, 'uniDINO.csv'), usecols=['path'])
with open(os.path.join(res_dir, 'SI-SimCLR', 'latent_embeddings_train.pickle'), 'rb') as f:
    raw_emb = pickle.load(f)
assert len(raw_emb) == len(uni_order), 'SI-SimCLR pickle and uniDINO.csv should have the same number of rows'
uni_order['row'] = np.arange(len(uni_order))
rows = align_on_annotations(uni_order, 'path')['row'].values
raw_emb = raw_emb[rows]
scaled_emb = StandardScaler().fit_transform(raw_emb)
UMAP_features['Augmented\nSimCLR'] = run_umap(scaled_emb)
PCA_features['Augmented\nSimCLR'] = pca.fit_transform(scaled_emb)
print('Augmented SimCLR done')

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


# Define the labels for coloring
labels = ['cell_line', 'mutation', 'condition', 'experiment']
order = ['Hand-crafted', 'Pretrained\nResNet18', 'Pretrained\nResNet50', 'Standard\nSimCLR',
         'uniDINO', 'CellPaint SSL', 'Augmented\nSimCLR']
UMAP_features = {name: UMAP_features[name] for name in order if name in UMAP_features}
embeddings_keys = list(UMAP_features.keys())
palettes = {'cell_line': cell_line_palette, 'mutation': mutation_palette, 'condition': condition_palette, 'experiment': day_palette}


####################################################
# Figure: one row per method, one column per label
####################################################

fig, axes = plt.subplots(len(embeddings_keys), len(labels), figsize=(20, 5 * len(embeddings_keys)),
                         constrained_layout=True, squeeze=False)
print(axes.shape)

for i, embedding_key in enumerate(embeddings_keys):  # only the methods that were computed
    for j, label in enumerate(labels):
        ax = axes[i, j]
        ax.scatter(
            UMAP_features[embedding_key][:, 0],
            UMAP_features[embedding_key][:, 1],
            c=global_annotations[label].map(palettes[label]).fillna('#808080'),  # CTRL3 has no colour -> grey
            s=4,
            alpha=0.4,
            rasterized=True,
        )
        if j == 0:
            ax.text(-0.1, 0.5, embedding_key, va='center', ha='right', rotation=90, transform=ax.transAxes, fontsize=28, fontweight='bold')
        ax.axis('off')

out_pdf = os.path.join(res_dir, 'all_UMAP_zscore.pdf')
plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
plt.savefig(out_pdf.replace('.pdf', '.png'), dpi=120, bbox_inches='tight')
print('Saved', out_pdf)
