#!/usr/bin/env python3
import argparse
import sys
import os
import pickle
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler

# Optional PyTorch implementation
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# project-specific utils
sys.path.append('../')
from utils.mypath import MyPath
from utils.supervised_classification import (
    PyTorchLogisticRegression, 
    train_torch_model
)

def run_leave_one_batch_out(global_annotations, features, ds_seeds, model_seeds, args, device):
    """
    Core framework: For each feature set and task, rotate through batches.
    Train on (N-1) batches, test on the held-out batch.
    
    IMPROVEMENT: Ensures strict balancing where every batch contributes 
    exactly the same number of samples per class (determined by the 
    global minimum across all batches in the task).
    """
    tasks = {
        "ALS": {"label_col": "als_label", "pos_label": "als", 
                "filter": lambda df: df[df['condition'] == 'untreated']},
        "TIC high": {"label_col": "condition", "pos_label": "A1 high", 
                     "filter": lambda df: df[(df['als_label'] == 'control') & df['condition'].isin(['untreated', 'A1 high'])]},
        "TIC low": {"label_col": "condition", "pos_label": "A1 low", 
                    "filter": lambda df: df[(df['als_label'] == 'control') & df['condition'].isin(['untreated', 'A1 low'])]},
    }

    results = []
    
    # Ensure global_annotations has a unique index that aligns with the embeddings
    # (Assuming row i in global_annotations corresponds to row i in embeddings)
    if not global_annotations.index.is_unique:
        global_annotations = global_annotations.reset_index(drop=True)

    for name, emb in tqdm(features.items(), desc="Feature Sets"):
        for task_name, info in tasks.items():
            label_col = info['label_col']
            pos_label = info['pos_label']
            
            # 1. Filter global data for this specific task
            # We keep the original index so we can slice 'emb' correctly later
            task_df = info['filter'](global_annotations)
            
            # Check which batches are valid (must have both classes)
            valid_batches = []
            for b in task_df['experiment'].unique():
                batch_subset = task_df[task_df['experiment'] == b]
                if batch_subset[label_col].nunique() == 2:
                    valid_batches.append(b)

            if len(valid_batches) < 2:
                continue

            # 2. Determine Global Minimum Class Count
            # To ensure every batch contributes equally, we find the smallest class count
            # observed in any of the valid batches.
            min_counts = []
            for b in valid_batches:
                batch_subset = task_df[task_df['experiment'] == b]
                min_counts.append(batch_subset[label_col].value_counts().min())
            
            # This is the target number of samples per class per batch
            n_per_class = min(min_counts)

            # 3. Leave-One-Batch-Out Loop
            for test_batch in valid_batches:
                for ds_seed in ds_seeds:
                    # Determine Train/Test Indices
                    train_indices_global = []
                    test_indices_global = []

                    rng = np.random.RandomState(ds_seed)

                    for b in valid_batches:
                        batch_subset = task_df[task_df['experiment'] == b]
                        unique_classes = batch_subset[label_col].unique()
                        
                        batch_chosen_indices = []
                        for cls in unique_classes:
                            # Get global indices of this class in this batch
                            cls_indices = batch_subset[batch_subset[label_col] == cls].index.values
                            
                            # Sample exactly n_per_class
                            chosen = rng.choice(cls_indices, n_per_class, replace=False)
                            batch_chosen_indices.extend(chosen)
                        
                        # Assign to Train or Test
                        if b == test_batch:
                            test_indices_global.extend(batch_chosen_indices)
                        else:
                            train_indices_global.extend(batch_chosen_indices)
                    
                    # 4. Prepare Data Arrays using Global Indices
                    # FIX: Using .values on the DataFrame index ensures we grab the correct rows from 'emb'
                    X_train = emb[train_indices_global]
                    y_train = global_annotations.loc[train_indices_global, label_col].values
                    
                    X_test = emb[test_indices_global]
                    y_test = global_annotations.loc[test_indices_global, label_col].values
                    img_codes_test = global_annotations.loc[test_indices_global, 'img_code'].values

                    # 5. Training Loop
                    for m_seed in model_seeds:
                        if args.model_type == 'mlp':
                            unique_labels = np.unique(y_train)
                            label_to_int = {str(l): i for i, l in enumerate(unique_labels)}
                            y_train_int = np.array([label_to_int[str(x)] for x in y_train])
                            
                            _, probs = train_torch_model(
                                X_train, y_train_int, X_test, device,
                                hidden_sizes=tuple(args.hidden_layer_sizes),
                                lr=args.lr, max_epochs=args.max_epochs, seed=m_seed
                            )
                            # Handle potential missing pos_label in unique_labels if batch is weird (unlikely with logic above)
                            if pos_label in unique_labels:
                                pos_idx = list(unique_labels).index(pos_label)
                                pos_proba = probs[:, pos_idx]
                                pred_classes = [unique_labels[int(p.argmax())] for p in probs]
                            else:
                                # Fallback if purely one class (should be caught by valid_batches logic)
                                pos_proba = np.zeros(len(probs))
                                pred_classes = [unique_labels[0]] * len(probs)
                        
                        else: # Logistic Regression
                            clf = PyTorchLogisticRegression(input_dim=X_train.shape[1], lr=args.lr)
                            clf.fit(X_train, y_train, max_iter=args.max_iter, seed=m_seed)
                            
                            y_proba = clf.predict_proba(X_test)
                            classes = list(clf.classes_)
                            
                            if pos_label in classes:
                                pos_idx = classes.index(pos_label)
                                pos_proba = y_proba[:, pos_idx]
                            else:
                                pos_proba = np.zeros(len(y_test)) # Should not happen

                            pred_classes = clf.predict(X_test)

                        results.append(pd.DataFrame({
                            'exp_code': name, 'classifier': task_name, 'test_batch': test_batch,
                            'ds_seed': ds_seed, 'model_seed': m_seed, 'true': list(y_test),
                            'pred_class': list(pred_classes), 'pred_prob': list(pos_proba),
                            'img_code': list(img_codes_test)
                        }))

    return results

def main(args):
    out_csv = args.out
    out_dir = os.path.dirname(out_csv)
    if out_dir: os.makedirs(out_dir, exist_ok=True)

    # 1. Feature Loading Logic
    global_annotations = pd.read_csv('../data/crop16_EF_2405.csv')
    global_annotations['file_name'] = global_annotations['img_code'].apply(lambda x: os.path.basename(x).split('.')[0])
    
    # Filter specific experiment/well conditions
    if 'E1' in global_annotations['experiment'].unique():
        global_annotations = global_annotations[~((global_annotations['fov'] == 5) & (global_annotations['well'] == 52) & (global_annotations['experiment'] == 'E1'))]

    global_annotations = global_annotations[global_annotations['mutation'] != 'SOD1']
    
    # Reset index to ensure 0..N alignment with embeddings arrays constructed below
    global_annotations = global_annotations.reset_index(drop=True)

    features = {}
    res_dir = '../embeddings'
    config_dir = '../configs'

    exp_codes = ['240517_11', '240813_5']
    exp_names = {'240517_11': 'Standard SimCLR',
                 '240813_5': 'SI SimCLR',}


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
                global_annotations[['file_name']],
                temp_annotations, 
                on='file_name', 
                how='inner'
            )

            # Ensure alignment
            assert temp_annotations['file_name'].tolist() == global_annotations['file_name'].tolist()

            if config['pairing_strategy'] == 'mut_cond':
                temp_annotations = temp_annotations.reset_index(drop=True)
            raw_emb = temp_annotations[[c for c in temp_annotations.columns if c.startswith('emb')]].values

            name = exp_names[e]
            scaled_emb = StandardScaler().fit_transform(raw_emb)
            features[name] = scaled_emb
            assert not np.isnan(scaled_emb).any()


    handcrafted_df = pd.read_csv(os.path.join(res_dir, 'handcrafted.csv'))
    if 'img_code' in handcrafted_df.columns:
        handcrafted_df['file_name'] = handcrafted_df['img_code'].apply(lambda x: os.path.basename(x).split('.')[0])

    handcrafted_df = pd.merge(
        global_annotations[['file_name']],
        handcrafted_df, 
        on='file_name', 
        how='inner'
    )

    assert handcrafted_df['file_name'].tolist() == global_annotations['file_name'].tolist()
    handcrafted_feat = handcrafted_df.drop(['img_code', 'file_name'], axis=1)
    handcrafted_feat = handcrafted_feat.dropna()
    handcrafted_feat = handcrafted_feat.loc[:, handcrafted_feat.var() != 0].values
    handcrafted_feat = StandardScaler().fit_transform(handcrafted_feat)
    features['hand-crafted'] = handcrafted_feat


    pretrain_resnet_annotations = pd.read_csv(os.path.join(res_dir, 'pretrained_resnet_embeddings.csv'))
    if 'img_code' in pretrain_resnet_annotations.columns:
        pretrain_resnet_annotations['file_name'] = pretrain_resnet_annotations['img_code'].apply(lambda x: os.path.basename(x).split('.')[0])
    elif 'simple_code' in pretrain_resnet_annotations.columns:
        pretrain_resnet_annotations.rename(columns={'simple_code': 'file_name'}, inplace=True)

    pretrain_resnet_annotations = pd.merge(
        global_annotations[['file_name']],
        pretrain_resnet_annotations, 
        on='file_name', 
        how='inner'
    )

    assert pretrain_resnet_annotations['file_name'].tolist() == global_annotations['file_name'].tolist()

    pretrained_emb = pretrain_resnet_annotations.drop([c for c in pretrain_resnet_annotations.columns if 'emb' not in c], axis=1)
    pretrained_emb = pretrained_emb.dropna()
    scaled_pretrained = StandardScaler().fit_transform(pretrained_emb)
    features['Pretrained\nResNet18'] = scaled_pretrained

    low_variance_threshold = 1e-5
    uni_emb = pd.read_csv(os.path.join(res_dir, 'uniDINO.csv'))
    if 'path' in uni_emb.columns:
        uni_emb['file_name'] = uni_emb['path'].apply(lambda x: os.path.basename(x).split('.')[0])

    uni_emb = pd.merge(
        global_annotations[['file_name']], 
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
    features['uniDINO'] = scaled_uni_highvar

    cellpaint_df = pd.read_csv(os.path.join(res_dir, 'CellPaint-SSL.csv'))
    if 'img_code' in cellpaint_df.columns:
        cellpaint_df['file_name'] = cellpaint_df['img_code'].apply(lambda x: os.path.basename(x).split('.')[0])

    cellpaint_df = pd.merge(
        global_annotations[['file_name']], 
        cellpaint_df, 
        on='file_name', 
        how='inner'
    )

    assert cellpaint_df['file_name'].tolist() == global_annotations['file_name'].tolist()
    cellpaint_emb = cellpaint_df.drop([c for c in cellpaint_df.columns if 'emb' not in c], axis=1)

    variances = cellpaint_emb.var(axis=0)
    high_variance_mask = variances > low_variance_threshold
    scaled_cellpaint_highvar = StandardScaler().fit_transform(cellpaint_emb.values[:, high_variance_mask])
    features['CellPaint SSL'] = scaled_cellpaint_highvar
    
    # 2. Setup Seeds
    ds_seeds = list(range(args.n_ds_seeds))
    model_seeds = list(range(args.n_model_seeds))

    # 3. Device selection
    device = torch.device('cuda' if (args.use_gpu and torch.cuda.is_available()) else 'cpu')
    
    # 4. Run LOBO
    print(f"Starting Leave-One-Batch-Out using {args.model_type.upper()}...")
    results_list = run_leave_one_batch_out(global_annotations, features, ds_seeds, model_seeds, args, device)
    
    if results_list:
        results_df = pd.concat(results_list, ignore_index=True)
        results_df.to_csv(out_csv, index=False)
        print(f'Done! Saved results to {out_csv}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Leave-One-Batch-Out classification')
    parser.add_argument('--out', type=str, default='results_lobo.csv')
    parser.add_argument('--model-type', type=str, choices=['mlp', 'logreg'], default='logreg')
    parser.add_argument('--n-ds-seeds', type=int, default=5, help='Seeds for downsampling/balancing')
    parser.add_argument('--n-model-seeds', type=int, default=5, help='Seeds for model initialization')
    parser.add_argument('--use-gpu', action='store_true')
    parser.add_argument('--hidden-layer-sizes', nargs='+', type=int, default=[100])
    parser.add_argument('--max-iter', type=int, default=100)
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1)
    args = parser.parse_args()

    main(args)