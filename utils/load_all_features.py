import sys
sys.path.append('../')
import os
import pickle
import yaml
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')
from utils.mypath import MyPath


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    
def load_all_features(ann_path=None):

    if ann_path is None:
        ann_path = os.path.join(REPO_ROOT, 'data', 'annotations_HDBSCAN_filter_no_adjacent_single.csv')

    global_annotations = pd.read_csv(ann_path)
    global_annotations['file_name'] = global_annotations['img_code'].apply(lambda x: os.path.basename(x).split('.')[0])

    if 'E1' in global_annotations['experiment'].unique():
        global_annotations = global_annotations[~((global_annotations['fov'] == 5) & (global_annotations['well'] == 52) & (global_annotations['experiment'] == 'E1'))]

    global_annotations = global_annotations[global_annotations['mutation'] != 'SOD1']

    features = {}
    res_dir = os.path.join(REPO_ROOT, 'embeddings')
    config_dir = os.path.join(REPO_ROOT, 'configs')

    exp_codes = ['240517_11', '240813_5']
    exp_names = {'240517_11': 'Standard SimCLR',
                 '240813_5': 'SI SimCLR',}

    # load simclr/24... experiments
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
            features[name] = scaled_emb
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
    features['hand-crafted'] = handcrafted_feat


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
    features['Pretrained\nResNet18'] = scaled_pretrained

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
    features['uniDINO'] = scaled_uni_highvar
    #UMAP_features['uniDINO'] = UMAP(n_components=2, random_state=0, n_neighbors=15).fit_transform(scaled_uni_highvar)
    #PCA_features['uniDINO'] = pca.fit_transform(scaled_uni_highvar)
    #PCA_var_explained['uniDINO'] = pca.explained_variance_ratio_

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
    features['CellPaint SSL'] = scaled_cellpaint_highvar

    return features, global_annotations