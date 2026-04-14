import pandas as pd
import os
import numpy as np
import pickle
from utils.mypath import MyPath
from utils.annotations_utils import add_well_column


def load_data(p):
    annotations_df = pd.read_csv(MyPath.db_root_dir(p['train_db_name']))
    if 'well' not in annotations_df.columns:
        annotations_df = add_well_column(annotations_df)
    if 'well_id' not in annotations_df.columns:
        annotations_df['well_id'] = annotations_df['experiment'] + '_' + annotations_df['plate'] + '_W' + str(annotations_df['well'])
    if 'exp_plate' not in annotations_df.columns:
        annotations_df['exp_plate'] = annotations_df['experiment'] + '_' + annotations_df['plate']
    raw_emb = pickle.load(open(os.path.join(p['target_dir'], 'latent_embeddings_train.pickle'), 'rb'))
    return annotations_df, raw_emb

def filter_data(p, annotations_df, raw_emb, single_plate=None):

    annotations_df = annotations_df[annotations_df['mutation']!='SOD1']
    df_shape = annotations_df.shape[0]
    if p['pairing_strategy'] == 'mut_cond':
        annotations_df.reset_index(drop=True, inplace=True)
    df_index = annotations_df.index

    if single_plate is not None:
        annotations_df = annotations_df[annotations_df['exp_plate'] == single_plate]

    df_index_after_filter = annotations_df.index
    raw_emb_shape = raw_emb.shape[0]
    raw_emb = raw_emb[annotations_df.index.to_numpy()]
    
    raw_emb_shape_after_filter = raw_emb.shape[0]
    
    return annotations_df, raw_emb


def annot_func(data):
    return np.vectorize(lambda x: f"{x:.0f}" if x != 0 else "")(data)