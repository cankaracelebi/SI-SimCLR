#!/usr/bin/env python3
import os, sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path[:0] = [REPO_ROOT, os.path.join(REPO_ROOT, 'SI-SimCLR')]

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

    for name, emb in tqdm(features.items(), desc="Feature Sets", position=0):
        label = name.replace('\n', ' ')
        # One tick per classifier fit; the total grows as each task's valid batches become known
        fit_bar = tqdm(total=0, desc=label, unit='fit', position=1, leave=False)
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
            fit_bar.total += len(valid_batches) * len(ds_seeds) * len(model_seeds)
            fit_bar.refresh()

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
                        fit_bar.set_postfix_str(f'{task_name}, test {test_batch}')
                        fit_bar.update(1)
        fit_bar.close()

    return results

def main(args):
    out_csv = args.out
    out_dir = os.path.dirname(out_csv)
    if out_dir: os.makedirs(out_dir, exist_ok=True)

    # 1. Feature Loading Logic
    global_annotations = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'crop16_EF_2405.csv'))
    global_annotations['file_name'] = global_annotations['img_code'].apply(lambda x: os.path.basename(x).split('.')[0])
    
    # Filter specific experiment/well conditions
    if 'E1' in global_annotations['experiment'].unique():
        global_annotations = global_annotations[~((global_annotations['fov'] == 5) & (global_annotations['well'] == 52) & (global_annotations['experiment'] == 'E1'))]

    global_annotations = global_annotations[global_annotations['mutation'] != 'SOD1']
    
    # Reset index to ensure 0..N alignment with embeddings arrays constructed below
    global_annotations = global_annotations.reset_index(drop=True)

    features = {}
    res_dir = os.path.join(REPO_ROOT, 'embeddings')
    config_dir = os.path.join(REPO_ROOT, 'configs')

    # Experiments saved as embeddings/<exp_code>/latent_embeddings_train.pickle.
    # Any that are not there yet (e.g. SimCLR runs still training) are skipped.
    exp_names = {'resnet50_imagenet': 'Pretrained\nResNet50',
                 'vits16_dino_imagenet': 'Pretrained\nViT-S',
                 '240517_11': 'Standard SimCLR',
                 '240813_5': 'SI SimCLR'}

    only = getattr(args, 'only', None)
    wanted = lambda key: not only or key in only

    for e, name in exp_names.items():
        if not wanted(e):
            continue
        emb_path = os.path.join(res_dir, e, 'latent_embeddings_train.pickle')
        if not os.path.exists(emb_path):
            print(f'Skipping {e}: {emb_path} not found', flush=True)
            continue
        config = yaml.safe_load(open(os.path.join(config_dir, e, 'simclr_acs.yml')))
        temp_annotations = pd.read_csv(MyPath.db_root_dir(config['train_db_name']))
        temp_annotations['file_name'] = temp_annotations['img_code'].apply(lambda x: os.path.basename(x).split('.')[0])

        with open(emb_path, 'rb') as f:
            og_emb = pickle.load(f)
        assert og_emb.shape[0] == len(temp_annotations)

        # Reorder embedding rows to follow global_annotations
        row_of = pd.Series(np.arange(len(temp_annotations)), index=temp_annotations['file_name'])
        raw_emb = og_emb[row_of.loc[global_annotations['file_name']].values]

        scaled_emb = StandardScaler().fit_transform(raw_emb)
        assert not np.isnan(scaled_emb).any()
        features[name] = scaled_emb
        print(f'Loaded {e}: {scaled_emb.shape}', flush=True)

    # Paper baselines distributed as CSVs (img_code/path + feature columns).
    # They are not in the repo or on Zenodo; each is used only if the file is present.
    def load_csv_baseline(file_name, emb_only=True, var_threshold=0.0):
        path = os.path.join(res_dir, file_name)
        if not os.path.exists(path):
            print(f'Skipping {file_name}: not found in {res_dir}', flush=True)
            return None
        df = pd.read_csv(path)
        print(f'Loaded {file_name}: {df.shape}', flush=True)
        for col in ['img_code', 'path']:
            if col in df.columns:
                df['file_name'] = df[col].apply(lambda x: os.path.basename(x).split('.')[0])
        df = pd.merge(global_annotations[['file_name']], df, on='file_name', how='inner')
        assert df['file_name'].tolist() == global_annotations['file_name'].tolist()
        if emb_only:
            feat = df[[c for c in df.columns if 'emb' in c]]
        else:
            feat = df.drop([c for c in ['img_code', 'path', 'file_name'] if c in df.columns], axis=1)
        feat = feat.loc[:, feat.var() > var_threshold].values
        return StandardScaler().fit_transform(feat)

    for name, file_name, kwargs in [
        ('hand-crafted', 'handcrafted.csv', dict(emb_only=False)),
        ('Pretrained\nResNet18', 'pretrained_resnet_embeddings.csv', dict()),
        ('uniDINO', 'uniDINO.csv', dict(var_threshold=1e-5)),
        ('CellPaint SSL', 'CellPaint-SSL.csv', dict(var_threshold=1e-5)),
    ]:
        if not wanted(file_name):
            continue
        feat = load_csv_baseline(file_name, **kwargs)
        if feat is not None:
            features[name] = feat

    assert features, 'No feature sets found'

    
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
    parser.add_argument('--only', nargs='+', default=None,
                        help='Restrict to these feature sets: experiment codes (e.g. vits16_dino_imagenet) '
                             'and/or baseline file names (e.g. handcrafted.csv). Default: all available.')
    args = parser.parse_args()

    main(args)