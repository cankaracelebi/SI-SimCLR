"""
Fig 2B/2C: best UMAP + KMeans ARI per embedding set, adapted from
embeddings_analysis/Fig_2BC_ARI_score.py (same protocol), with Pretrained ResNet50 and ViT-S (DINO, ImageNet) added.

Fig 2B (batch effect):         python experiments/resnet50/fig2b_ari.py
Fig 2C (ALS:TIC, per batch):   python experiments/resnet50/fig2c_ari.py

Options:
    --models resnet50 resnet18     only compute these models (default: all)
    --annotations <file in data/>  reference set of cells (default: Zenodo CSV)
    --n_jobs 4                     UMAP fits run in parallel (default: all cores)
    --quick                        single UMAP setting, for testing

Results are merged into results/analysis/<ARI_UMAP|ARI_UMAP_batch>/<annotations>/best_umap_kmeans_ari.yaml,
so models can be computed in separate runs.
"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(candidate for candidate in [SCRIPT_DIR, *SCRIPT_DIR.parents] if (candidate / 'configs' / 'env.yml').is_file())

sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import pickle
import os
import warnings
from umap import UMAP
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import itertools
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from utils.config import load_env_paths
from utils.utils import str2bool

from argparse import ArgumentParser

warnings.filterwarnings('ignore')

# --models key -> name used in the results yaml (same names as the reference results)
MODELS = {'resnet50': 'Pretrained\nResNet50',
          'vits': 'Pretrained\nViT-S',
          'simclr': 'Standard\nSimCLR',
          'si_simclr': 'Augmented\nSimCLR',
          'handcrafted': 'hand-crafted',
          'resnet18': 'Pretrained\nResNet18',
          'unidino': 'uniDINO',
          'cellpaint': 'CellPaint SSL'}

# use argparse to get TARGET_LABEL and BATCH_MODE
parser = ArgumentParser()
# Fig 2C labels: ALS/control x Basal/Low TIC/High TIC (K=4 or 6 depending on the batch)
parser.add_argument('--target_label', type=str, default='mut_cond')
parser.add_argument('--batch_mode', type=str2bool, default=True)
parser.add_argument('--models', nargs='+', choices=list(MODELS), default=list(MODELS))
parser.add_argument('--annotations', type=str, default='annotations_processed_crops.csv')
parser.add_argument('--n_jobs', type=int, default=-1)
parser.add_argument('--quick', action='store_true', help='single UMAP setting, for testing')
args = parser.parse_args()
TARGET_LABEL = args.target_label
BATCH_MODE = args.batch_mode
QUICK = args.quick

if TARGET_LABEL in ['experiment', 'batch', 'exp_plate']:
    assert not BATCH_MODE

ENV_PATHS = load_env_paths(str(REPO_ROOT / 'configs' / 'env.yml'))

# Same layout as the reference results read by Fig_2BC_ARI_boxplot.ipynb, one subfolder per set of cells
batch_target = TARGET_LABEL in ['experiment', 'batch', 'exp_plate']
output_dir = os.path.join(ENV_PATHS['analysis_results_root_dir'], 'ARI_UMAP_batch' if batch_target else 'ARI_UMAP',
                          Path(args.annotations).stem)
output_file = 'best_umap_kmeans_ari_quick.yaml' if QUICK else 'best_umap_kmeans_ari.yaml'
res_dir = ENV_PATHS['embeddings_root_dir']

os.makedirs(output_dir, exist_ok=True)


# Reference set and order of cells for every embedding set
global_annotations = pd.read_csv(os.path.join(REPO_ROOT, 'data', args.annotations))
global_annotations['file_name'] = global_annotations['img_code'].apply(lambda x: os.path.basename(x).split('.')[0])
global_annotations['mut_cond'] = global_annotations['mutation'] + '_' + global_annotations['condition']
if 'E1' in global_annotations['experiment'].unique():
    global_annotations = global_annotations[~((global_annotations['fov'] == 5) & (global_annotations['well'] == 52) & (global_annotations['experiment'] == 'E1'))]

global_annotations = global_annotations[global_annotations['mutation'] != 'SOD1']

assert TARGET_LABEL in global_annotations.columns

print("="*40)
print(f"Global annotations shape: {global_annotations.shape}")
print(f"First file name: {global_annotations['file_name'].iloc[0]}")
print("="*40)

# Hyperparameter grid for UMAP
n_neighbors_list = [15, 50, 100, 150, 200, 300, 600]
min_dist_list = [0.001, 0.1]
if QUICK:
    n_neighbors_list = [15]
    min_dist_list = [0.1]

embedding_data = {}

# Embeddings extracted by experiments/resnet50/extract_resnet50.py (same pipeline for ViT-S)
for key, exp_code in [('resnet50', 'resnet50_imagenet'), ('vits', 'vits16_dino_imagenet')]:
    if key not in args.models:
        continue
    # The pickle holds no image names: rows follow the Zenodo CSV it was extracted from
    zenodo_annotations = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'annotations_processed_crops.csv'))
    zenodo_annotations['file_name'] = zenodo_annotations['img_code'].apply(lambda x: os.path.basename(x).split('.')[0])

    with open(os.path.join(res_dir, exp_code, 'latent_embeddings_train.pickle'), 'rb') as f:
        og_emb = pickle.load(f)

    assert og_emb.shape[0] == len(zenodo_annotations)

    emb_df = pd.DataFrame(og_emb, columns=[f'emb{i}' for i in range(og_emb.shape[1])])
    model_df = pd.concat([zenodo_annotations[['file_name']], emb_df], axis=1)

    model_df = pd.merge(
        global_annotations[['file_name']], # The "template" order
        model_df,
        on='file_name',
        how='inner'
    )

    assert model_df['file_name'].tolist() == global_annotations['file_name'].tolist()

    raw_emb = model_df[[c for c in model_df.columns if c.startswith('emb')]].values
    scaled_emb = StandardScaler().fit_transform(raw_emb)
    # check that no NaNs are in embeddings
    assert not np.isnan(scaled_emb).any()
    embedding_data[MODELS[key]] = scaled_emb


# SimCLR embeddings sent as CSV (img_code + emb columns), same preprocessing as the original SimCLR pickles
for key, csv_name in [('simclr', 'SimCLR.csv'), ('si_simclr', 'SI_SimCLR.csv')]:
    if key not in args.models:
        continue

    simclr_df = pd.read_csv(os.path.join(res_dir, csv_name))
    simclr_df['file_name'] = simclr_df['img_code'].apply(lambda x: os.path.basename(x).split('.')[0])

    simclr_df = pd.merge(
        global_annotations[['file_name']], # The "template" order
        simclr_df,
        on='file_name',
        how='inner'
    )

    assert simclr_df['file_name'].tolist() == global_annotations['file_name'].tolist()

    raw_emb = simclr_df[[c for c in simclr_df.columns if c.startswith('emb')]].values
    scaled_emb = StandardScaler().fit_transform(raw_emb)
    assert not np.isnan(scaled_emb).any()
    embedding_data[MODELS[key]] = scaled_emb


if 'handcrafted' in args.models:
    handcrafted_df = pd.read_csv(os.path.join(res_dir, 'handcrafted.csv'))
    if 'img_code' in handcrafted_df.columns:
        handcrafted_df['file_name'] = handcrafted_df['img_code'].apply(lambda x: os.path.basename(x).split('.')[0])

    handcrafted_df = pd.merge(
        global_annotations[['file_name']], # The "template" order
        handcrafted_df,
        on='file_name',
        how='inner'
    )

    assert handcrafted_df['file_name'].tolist() == global_annotations['file_name'].tolist()
    handcrafted_feat = handcrafted_df.drop(['img_code', 'file_name'], axis=1)
    handcrafted_feat = handcrafted_feat.dropna()
    handcrafted_feat = handcrafted_feat.loc[:, handcrafted_feat.var() != 0].values
    handcrafted_feat = StandardScaler().fit_transform(handcrafted_feat)
    embedding_data[MODELS['handcrafted']] = handcrafted_feat


if 'resnet18' in args.models:
    pretrain_resnet_annotations = pd.read_csv(os.path.join(res_dir, 'pretrained_resnet_embeddings.csv'))
    if 'img_code' in pretrain_resnet_annotations.columns:
        pretrain_resnet_annotations['file_name'] = pretrain_resnet_annotations['img_code'].apply(lambda x: os.path.basename(x).split('.')[0])
    elif 'simple_code' in pretrain_resnet_annotations.columns:
        pretrain_resnet_annotations.rename(columns={'simple_code': 'file_name'}, inplace=True)

    pretrain_resnet_annotations = pd.merge(
        global_annotations[['file_name']], # The "template" order
        pretrain_resnet_annotations,
        on='file_name',
        how='inner'
    )

    assert pretrain_resnet_annotations['file_name'].tolist() == global_annotations['file_name'].tolist()

    pretrained_emb = pretrain_resnet_annotations.drop([c for c in pretrain_resnet_annotations.columns if 'emb' not in c], axis=1)
    pretrained_emb = pretrained_emb.dropna()
    scaled_pretrained = StandardScaler().fit_transform(pretrained_emb)
    embedding_data[MODELS['resnet18']] = scaled_pretrained

low_variance_threshold = 1e-5

if 'unidino' in args.models:
    uni_emb = pd.read_csv(os.path.join(res_dir, 'uniDINO.csv'), low_memory=False)
    if 'path' in uni_emb.columns:
        uni_emb['file_name'] = uni_emb['path'].apply(lambda x: os.path.basename(x).split('.')[0])

    uni_emb = pd.merge(
        global_annotations[['file_name']], # The "template" order
        uni_emb,
        on='file_name',
        how='inner'
    )
    assert uni_emb['file_name'].tolist() == global_annotations['file_name'].tolist()

    uni_emb_feat = uni_emb.drop([c for c in uni_emb.columns if 'emb' not in c], axis=1)
    variances = uni_emb_feat.var(axis=0)
    high_variance_mask = variances > low_variance_threshold
    print(f"uniDINO: {len(high_variance_mask)} features, {high_variance_mask.sum()} high variance", flush=True)
    scaled_uni_highvar = StandardScaler().fit_transform(uni_emb_feat.values[:, high_variance_mask])
    embedding_data[MODELS['unidino']] = scaled_uni_highvar

if 'cellpaint' in args.models:
    cellpaint_df = pd.read_csv(os.path.join(res_dir, 'CellPaint-SSL.csv'))
    if 'img_code' in cellpaint_df.columns:
        cellpaint_df['file_name'] = cellpaint_df['img_code'].apply(lambda x: os.path.basename(x).split('.')[0])

    cellpaint_df = pd.merge(
        global_annotations[['file_name']], # The "template" order
        cellpaint_df,
        on='file_name',
        how='inner'
    )

    assert cellpaint_df['file_name'].tolist() == global_annotations['file_name'].tolist()
    cellpaint_emb = cellpaint_df.drop([c for c in cellpaint_df.columns if 'emb' not in c], axis=1)

    variances = cellpaint_emb.var(axis=0)
    high_variance_mask = variances > low_variance_threshold
    scaled_cellpaint_highvar = StandardScaler().fit_transform(cellpaint_emb.values[:, high_variance_mask])
    embedding_data[MODELS['cellpaint']] = scaled_cellpaint_highvar

# --- PROCESSING ---

def umap_kmeans_ari(features, labels, k, nn, md):
    reducer = UMAP(n_neighbors=nn, min_dist=md, n_components=2, random_state=42)
    u_emb = reducer.fit_transform(features)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(u_emb)

    return adjusted_rand_score(labels, clusters)

# Define the iterations
batches = global_annotations['experiment'].unique() if BATCH_MODE else ['all']
grid = list(itertools.product(n_neighbors_list, min_dist_list))

# One task per (model, batch, UMAP setting), all run in parallel
slices = {}
for name, features in embedding_data.items():
    for batch in batches:
        if BATCH_MODE:
            mask = (global_annotations['experiment'] == batch).values
            current_feat = features[mask]
            current_labels = global_annotations.loc[mask, TARGET_LABEL].values
        else:
            current_feat = features
            current_labels = global_annotations[TARGET_LABEL].values

        # Dynamic K: Number of unique labels present in THIS specific slice
        current_k = len(np.unique(current_labels))

        if current_k < 2:
            print(f"  Skipping {name} batch {batch}: only {current_k} class present.")
            continue
        slices[(name, batch)] = (current_feat, current_labels, current_k)

# Longest fits first (cost grows with n_neighbors, cells and dimensions) for better load balancing
tasks = sorted(((key, nn, md) for key in slices for nn, md in grid),
               key=lambda t: -t[1] * slices[t[0]][0].shape[0] * slices[t[0]][0].shape[1])
print(f"{len(tasks)} UMAP fits for {len(embedding_data)} models, n_jobs={args.n_jobs}", flush=True)

aris = Parallel(n_jobs=args.n_jobs, verbose=10)(
    delayed(umap_kmeans_ari)(*slices[key], nn, md) for key, nn, md in tasks
)
aris = {task: ari for task, ari in zip(tasks, aris)}

# Dictionary to store the best results per embedding set
best_results = {}

for (name, batch), (_, _, current_k) in slices.items():
    best_ari = -1
    best_params = {}

    # Same selection as the serial loop: first best setting in grid order
    for nn, md in grid:
        ari = aris[((name, batch), nn, md)]
        if ari > best_ari:
            best_ari = ari
            best_params = {
                'n_neighbors': int(nn), # Cast to int for YAML serialization
                'min_dist': float(md),
                'ari': float(ari),
                'k_used': int(current_k)
            }

    best_results.setdefault(name, {})[str(batch)] = best_params
    print(f"  {name.replace(chr(10), ' ')} | batch {batch} best ARI: {best_ari:.4f} (nn={best_params['n_neighbors']}, md={best_params['min_dist']})", flush=True)

# save as yaml file, keeping models computed in previous runs
output_path = os.path.join(output_dir, output_file)
if os.path.exists(output_path):
    with open(output_path, 'r') as f:
        previous_results = yaml.safe_load(f) or {}
    previous_results.update(best_results)
    best_results = previous_results

with open(output_path, 'w') as f:
    yaml.dump(best_results, f)

print(f"\nOptimization Complete. Results saved to {output_path}")
