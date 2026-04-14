import sys
sys.path.append('../')

import numpy as np
import pandas as pd
import pickle
import os
import warnings
from sklearn.preprocessing import StandardScaler
from utils.config import create_config
from utils.mypath import MyPath
  
warnings.filterwarnings('ignore')

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_CONFIG_DIR = os.path.join(REPO_ROOT, 'configs')


def fix_embedding_format(embedding_str):
    try:
        # Replace spaces with commas inside the brackets
        fixed_str = embedding_str.replace(' ', ',')
        return np.array(eval(fixed_str))
    except Exception as e:
        print(f"Error processing embedding: {embedding_str} - {e}")
        return None



def setup_experiment(x: str, single_exps: list=['E1', 'E2', 'E3', 'E4'], source_dir=DEFAULT_CONFIG_DIR):
    # Directory and experiment setup
    env = os.path.join(source_dir, 'env.yml')
    hypp = os.path.join(source_dir, x, 'simclr_acs.yml')
    p = create_config(env, hypp)

    annotations_df = pd.read_csv(MyPath.db_root_dir(p['train_db_name']))
    raw_emb = pickle.load(open(os.path.join(p['emb_dir'], 'latent_embeddings_train.pickle'), 'rb'))

    if single_exps is not None:
        annotations_df = annotations_df[annotations_df['experiment'].isin(single_exps)]

    if 'E1' in annotations_df['experiment'].unique():
        annotations_df = annotations_df[~((annotations_df['fov'] == 5) & (annotations_df['well'] == 52) & (annotations_df['experiment'] == 'E1'))]

    annotations_df = annotations_df[annotations_df['mutation'] != 'SOD1']
    raw_emb = raw_emb[annotations_df.index]
    annotations_df['mut_cond'] = annotations_df['mutation'] + '_' + annotations_df['condition']
    annotations_df.reset_index(drop=True, inplace=True)

    if p.get('image_root_dir') is not None:
        annotations_df['img_code'] = annotations_df['img_code'].apply(
            lambda x: x if os.path.isabs(x) else os.path.join(p['image_root_dir'], os.path.basename(x))
        )
    
    return annotations_df, raw_emb


def load_process_handcrafted_features(annotations_df,
                                      process=True,
                                      source_dir=None,
                                      img_source_dir=None):

    if source_dir is None:
        source_dir = os.path.join(REPO_ROOT, 'embeddings')

    handcrafted_path = os.path.join(source_dir, 'classical-features.csv')
    if not os.path.exists(handcrafted_path):
        handcrafted_path = os.path.join(source_dir, 'handcrafted.csv')

    if not os.path.exists(handcrafted_path):
        raise FileNotFoundError(
            f"Could not find classical-features.csv or handcrafted.csv in {source_dir}"
        )
    
    hand_crafted_annotations = pd.read_csv(handcrafted_path)

    if img_source_dir is None and len(annotations_df) > 0:
        sample_img = annotations_df['img_code'].iloc[0]
        if isinstance(sample_img, str) and os.path.isabs(sample_img):
            img_source_dir = os.path.dirname(sample_img)

    if img_source_dir is not None:
        hand_crafted_annotations['img_code'] = hand_crafted_annotations['img_code'].apply(
            lambda x: os.path.join(
                img_source_dir,
                os.path.splitext(os.path.basename(str(x)))[0] + '.tif'
            )
        )
    hand_crafted_annotations = hand_crafted_annotations[hand_crafted_annotations['img_code'].isin(annotations_df['img_code'])]
    
    hand_crafted_annotations.reset_index(drop=True, inplace=True)

    nan_columns = hand_crafted_annotations.columns[hand_crafted_annotations.isna().any()].tolist()
    hand_crafted_annotations = hand_crafted_annotations.drop(columns=nan_columns)

    # Remove columns with 0 variance
    variance = hand_crafted_annotations.drop(columns=['img_code']).var()
    zero_variance_columns = variance[variance == 0].index.tolist()
    hand_crafted_annotations = hand_crafted_annotations.drop(columns=zero_variance_columns)
    hc_features = hand_crafted_annotations.drop(columns=['img_code']).values
    if process:
        scaler = StandardScaler()
        hc_features = scaler.fit_transform(hc_features)

    return hand_crafted_annotations, hc_features



def load_resnet_pretrained_features(annotations_df,
                                    source_dir=os.path.join(REPO_ROOT, 'results', '250411_pt_resnet18')):
    
    pretrain_resnet_annotations = pd.read_csv(os.path.join(source_dir, 'pretrained_resnet_embeddings.csv'))
    pretrain_resnet_annotations = pretrain_resnet_annotations[pretrain_resnet_annotations['img_code'].isin(annotations_df['img_code'])]
    pretrain_resnet_annotations = pretrain_resnet_annotations.set_index('img_code').reindex(annotations_df['img_code']).reset_index()
    
    pretrain_resnet_annotations['parsed_embedding'] = pretrain_resnet_annotations['pretrained_resnet_embedding'].apply(fix_embedding_format)
    pretrained_emb = np.stack(pretrain_resnet_annotations['parsed_embedding'].values)
    #print('Shape of pretrained embeddings:', pretrained_emb.shape)

    return pretrain_resnet_annotations, pretrained_emb
