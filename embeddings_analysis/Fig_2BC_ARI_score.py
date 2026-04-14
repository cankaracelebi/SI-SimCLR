import sys
sys.path.append('../')
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import os
import shutil
import math
from PIL import Image
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['legend.markerscale'] = 1.5
mpl.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from umap import UMAP
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import itertools
from sklearn.preprocessing import StandardScaler
from utils.mypath import MyPath
from utils.mypalette import mutation_palette, condition_palette, day_palette, cell_line_palette
from utils.config import load_env_paths
from utils.snn import *
from utils.start_notebook import setup_experiment, load_process_handcrafted_features, fix_embedding_format
from utils.utils import str2bool
from sklearnex import patch_sklearn
patch_sklearn()

from argparse import ArgumentParser, ArgumentTypeError
  
warnings.filterwarnings('ignore')


# use argparse to get TARGET_LABEL and BATCH_MODE
parser = ArgumentParser()
parser.add_argument('--target_label', type=str, default='mutation')
parser.add_argument('--batch_mode', type=str2bool, default=True)
args = parser.parse_args()
TARGET_LABEL = args.target_label
BATCH_MODE = args.batch_mode

if TARGET_LABEL in ['experiment', 'batch', 'exp_plate']:
    assert not BATCH_MODE

#single_exp = ['E1', 'E2', 'E3', 'E4']
exp_codes = ['hand-crafted', 'pretrained resnet18', '240517_11', '240813_5', 'uniDINO_crop_nopostproc']
exp_names = {'240517_11': 'Standard\nSimCLR',
             '240813_5': 'Augmented\nSimCLR',}

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(candidate for candidate in [SCRIPT_DIR, *SCRIPT_DIR.parents] if (candidate / 'configs' / 'env.yml').is_file())
ENV_PATHS = load_env_paths(str(REPO_ROOT / 'configs' / 'env.yml'))

output_dir = os.path.join(ENV_PATHS['analysis_results_root_dir'], 'ARI_UMAP')
res_dir = ENV_PATHS['embeddings_root_dir']
config_dir = os.path.join(REPO_ROOT, 'configs')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#order = ['Hand crafted', 'Pretrained ResNet18', 'Standard', 'Adjacent Neighbors', 'UNI inference', 'UNI full', 'UNI explora']
order = ['Hand-crafted', 'Pretrained ResNet18', 'Standard SimCLR', 'uniDINO', 'CellPaint SSL', 'Augmented SimCLR']
#days = ['E1', 'E2', 'E3', 'E4']

#img_source_dir = '/path/to/image/folder'
global_annotations = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'annotations_HDBSCAN_filter_no_adjacent_single.csv'))
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
#n_neighbors_list = [15, 200]
min_dist_list = [0.001, 0.1]

# Dictionary to store the best results per embedding set
best_results = {}

#UMAP_features = {}

embedding_data = {}

for e in exp_codes:
    if e.__contains__('24'):
        config = yaml.safe_load(open(os.path.join(config_dir, e, 'simclr_acs.yml')))
        temp_annotations = pd.read_csv(os.path.join(MyPath.db_root_dir(config['train_db_name'])))

        temp_annotations['file_name'] = temp_annotations['img_code'].apply(lambda x: os.path.basename(x).split('.')[0])

        with open(os.path.join(res_dir, e, 'latent_embeddings_train.pickle'), 'rb') as f:
            og_emb = pickle.load(f)
            
        assert og_emb.shape[0] == len(temp_annotations)

        for i in range(og_emb.shape[1]):
            temp_annotations[f'emb{i}'] = og_emb[:, i]

        temp_annotations = pd.merge(
            global_annotations[['file_name']], # The "template" order
            temp_annotations, 
            on='file_name', 
            how='inner'
        )

        assert temp_annotations['file_name'].tolist() == global_annotations['file_name'].tolist()

        if config['pairing_strategy'] == 'mut_cond':
            temp_annotations = temp_annotations.reset_index(drop=True)
        raw_emb = temp_annotations[[c for c in temp_annotations.columns if c.startswith('emb')]].values

        name = exp_names[e]
        scaled_emb = StandardScaler().fit_transform(raw_emb)
        embedding_data[name] = scaled_emb
        # check that no NaNs are in embeddings
        assert not np.isnan(scaled_emb).any()


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
#handcrafted_feat = handcrafted_df.drop(['img_code', 'mutation', 'condition', 'experiment', 'file_name', 'als_label'], axis=1)
handcrafted_feat = handcrafted_df.drop(['img_code', 'file_name'], axis=1)
handcrafted_feat = handcrafted_feat.dropna()
handcrafted_feat = handcrafted_feat.loc[:, handcrafted_feat.var() != 0].values
handcrafted_feat = StandardScaler().fit_transform(handcrafted_feat)
embedding_data['hand-crafted'] = handcrafted_feat


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
#UMAP_features['pretrained resnet18'] = pretrained_emb
pretrained_emb = pretrained_emb.dropna()
scaled_pretrained = StandardScaler().fit_transform(pretrained_emb)
embedding_data['Pretrained\nResNet18'] = scaled_pretrained

low_variance_threshold = 1e-5
uni_emb = pd.read_csv(os.path.join(res_dir, 'uniDINO.csv'))
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
embedding_data['uniDINO'] = scaled_uni_highvar

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
embedding_data['CellPaint SSL'] = scaled_cellpaint_highvar

embeddings_keys = list(embedding_data.keys())

# --- PROCESSING ---

for name, features in embedding_data.items():
    print(f"Optimizing {name}...", flush=True)
    
    # Define the iterations
    batches = global_annotations['experiment'].unique() if BATCH_MODE else ['all']
    batch_best_scores = {} # Changed to dict for better YAML readability

    for batch in batches:
        if BATCH_MODE:
            mask = global_annotations['experiment'] == batch
            current_feat = features[mask]
            current_labels = global_annotations.loc[mask, TARGET_LABEL]
        else:
            current_feat = features
            current_labels = global_annotations[TARGET_LABEL]

        # Dynamic K: Number of unique labels present in THIS specific slice
        current_k = current_labels.nunique()
        
        if current_k < 2:
            print(f"  Skipping Batch {batch}: only {current_k} class present.")
            continue

        best_ari = -1
        best_params = {}

        for nn, md in itertools.product(n_neighbors_list, min_dist_list):
            reducer = UMAP(n_neighbors=nn, min_dist=md, n_components=2, random_state=42)
            u_emb = reducer.fit_transform(current_feat)
            
            kmeans = KMeans(n_clusters=current_k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(u_emb)
            
            ari = adjusted_rand_score(current_labels, clusters)
            
            if ari > best_ari:
                best_ari = ari
                best_params = {
                    'n_neighbors': int(nn), # Cast to int for YAML serialization
                    'min_dist': float(md), 
                    'ari': float(ari),
                    'k_used': int(current_k)
                }
        
        batch_best_scores[batch] = best_params
        print(f"  Batch {batch} Best ARI: {best_ari:.4f} (nn={best_params['n_neighbors']}, md={best_params['min_dist']})", flush=True)

    best_results[name] = batch_best_scores

# save as yaml file
with open(os.path.join(output_dir, 'best_umap_kmeans_ari.yaml'), 'w') as f:
    yaml.dump(best_results, f)

print("\nOptimization Complete. Results saved.")