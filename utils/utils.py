"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
Functions added by Elisa Messori
"""
from argparse import ArgumentTypeError
import errno
import os
import pickle

#import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.spatial.distance import correlation, pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import random
import yaml
from easydict import EasyDict


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def scientific_format(pval):
    return f'{pval:.1e}'  # Format p-value as scientific notation (2 decimal places)


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)



class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


@torch.no_grad()
def fill_memory_bank(loader, model, memory_bank):
    model.eval()
    memory_bank.reset()

    for i, batch in enumerate(loader):
        images = batch['image'].cuda(non_blocking=True)
        targets = batch['target'].cuda(non_blocking=True)
        output = model(images)
        memory_bank.update(output, targets)
        if i % 100 == 0:
            print('Fill Memory Bank [%d/%d]' %(i, len(loader)))


def get_truncated_model(model):
    """
    Remove the projection head s.t. the base encoder can be used to retrieve embeddings/hidden features
    """
    truncated_model = nn.Sequential(*list(model.children())[:-1])
    return truncated_model


def average_euclidean_distance(x, y):
    return np.sqrt((sum((x-y)**2))/x.shape[0])


def custom_cartesian_product(l1, l2):
    result = []
    for i in l1:
        for j in l2:
            if i!=j:
                result.append((i, j))
    return result


def build_distance_matrix(base_dir: str, measure=average_euclidean_distance):
    """
    embeddings_array: ndarray where samples are sorted based on their index
    WARNING: samples passed as ndarray should be sorted based on their index if we want
    rows and columns of the distance matrix to be the actual indices of the samples for
    which distance is computed. 
    measure is a measure function among the ones defined for scipy function 'pdist';
    however, custom functions can also be passed (e.g. average_euclidean_distance)
    """
    embeddings_array = load_embeddings_from_array(base_dir, False)
    embeddings_array = StandardScaler().fit_transform(embeddings_array)
    distance_matrix = squareform(pdist(embeddings_array, measure))
    with open(os.path.join(base_dir, 'distance_matrix.pickle'), 'wb') as handle:
        pickle.dump(distance_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return distance_matrix


def load_distance_matrix(base_dir: str) -> np.ndarray:
    return pickle.load(open(os.path.join(base_dir, 'distance_matrix.pickle'), "rb"))


def load_embeddings_from_array(base_dir: str, PCA_: bool, projected = False) -> np.ndarray :
    """
    To be used if embeddings are stored as a numpy array. In this case, the index of each
    sample corresponds to the row of the embedding and the order is important to preserve
    this correspondence. 
    PCA is to be used to indicate whether you want to retrieve raw or reduced embeddings
    base_dir: directory where results of this specific experiment are stored.
    Returns the ndarray containing embeddings
    """
    if PCA_ and projected:
        raise ValueError("We currently do not have PCA for projected embeddings.")
    if not projected:
        if not PCA_:
            emb = pickle.load(open(os.path.join(base_dir, 'latent_embeddings.pickle'), "rb"))
        else:
            emb = pickle.load(open(os.path.join(base_dir, 'PCA_embeddings.pickle'), "rb"))
    else:
        if not PCA_:
            emb = pickle.load(open(os.path.join(base_dir, 'projected_embeddings.pickle'), "rb"))
    
    return emb


def convert_labels(label_words: list, legend: dict) -> list:
    """
    label_words: list of label strings (from pandas df)
    legend: dictionary indicating which string should be associated to which number
    Convert list of labels (in string format) to list of labels in int format
    The output list can be used to color scatterplots
    """
    return list(map(legend.get, label_words,))


def compute_entropy(embeddings_: np.ndarray) -> float:
    pca_entropy = PCA()
    pca_entropy.fit(embeddings_)
    S_pca= pca_entropy.singular_values_
    eigen_val_pca = S_pca**2 / np.sum(S_pca**2)
    entropy_pca = -1*np.sum(eigen_val_pca*np.log10(eigen_val_pca))/np.log10(len(eigen_val_pca))
    return entropy_pca


def compute_PCA(base_dir: str, projected: bool) -> np.ndarray:
    if projected:
        emb = pickle.load(open(os.path.join(base_dir, 'PCA_embeddings.pickle'), "rb"))
    else:
        emb = pickle.load(open(os.path.join(base_dir, 'latent_embeddings.pickle'), "rb"))
    pca = PCA(.95, copy=False)
    scaled_embeddings = StandardScaler().fit_transform(emb)
    reduced_embeddings = pca.fit_transform(scaled_embeddings)
    if projected:
        with open(os.path.join(base_dir, 'PCA_proj_embeddings.pickle'), 'wb') as handle:
            pickle.dump(reduced_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return reduced_embeddings
    

def experiments_summary(p):
    """
    Takes as input the config created at the beginning of training.
    It saves a .csv file to store general information about the experiments stored so far.
    """

    if os.path.exists(os.path.join(os.path.dirname(p['base_dir']), 'exp_summary.csv')):
        summary_df = pd.read_csv(os.path.join(os.path.dirname(p['base_dir']), 'exp_summary.csv'))
        # If the summary is present, we can just upload it and update it by adding a row relative to the current experiment
        current_exp = p['exp_code']
        if not current_exp in summary_df['exp codes'].to_list():
            path = os.path.join(p['base_dir'], 'simclr_acs.yml')
            with open(path, 'r') as stream:
                config = yaml.safe_load(stream)
            cfg = EasyDict()
            # Copy
            for k, v in config.items():
                if not k in ['model_kwargs', 'criterion', 'criterion_kwargs', 'optimizer', 'optimizer_kwargs', 'scheduler', 'scheduler_kwargs', 'num_workers',
                             'augmentation_kwargs', 'transformation_kwargs', 'exp_code', 'val_db_name']:
                    cfg[k] = v
                if k == 'augmentation_kwargs':
                    if type(config[k]) == dict:
                        for k1, v1 in config[k].items():
                            if not k1 in ['color_jitter', 'normalize']:
                                #key0 = list(v1.keys)[0]
                                cfg[f'{k1}_p'] = v1['p']
                            if (k1 == 'resize') and (v1['p']>0):
                                    cfg['img_size'] = v1['size']
                if k == 'criterion_kwargs':
                    for k2, v2 in config[k].items():
                        cfg['temperature'] = v2
                if k == 'model_kwargs':
                    for k3, v3 in config[k].items():
                        if k3 == 'features_dim':
                            cfg['proj_emb_size'] = v3
            # AT THE END OF THE LOOP I HAVE A DICTIONARY WITH THE INFO I WANT (NEED TO ADD TRAINED AND RESULTS INFO)
            cfg['exp codes'] = current_exp
            cfg['trained'] = 0
            cfg['results_computed'] = 0
            summary_df = pd.concat([summary_df, pd.DataFrame([cfg])], ignore_index=True, axis=0)
            #summary_df = summary_df.append(cfg, ignore_index=True)

    else: # If the summary is not present, then it is constructed considering also the current experiment
        base_parent = os.path.dirname(p['base_dir'])
        target_parent = os.path.dirname(p['target_dir'])
        all_dirs = os.listdir(base_parent)
        all_exps = [direc for direc in all_dirs if not direc.endswith('.csv')]
        hypp_dictionaries = []
        results_computed = []
        trained = []
        for exp in all_exps:
            path = os.path.join(base_parent, exp, 'simclr_acs.yml')
            with open(path, 'r') as stream:
                config = yaml.safe_load(stream)
            cfg = EasyDict()
            # Copy
            for k, v in config.items():
                if not k in ['model_kwargs', 'criterion', 'criterion_kwargs', 'optimizer', 'optimizer_kwargs', 'scheduler', 'scheduler_kwargs', 'num_workers',
                             'augmentation_kwargs', 'transformation_kwargs', 'exp_code', 'val_db_name']:
                    cfg[k] = v
                if k == 'augmentation_kwargs':
                    if type(config[k]) == dict:
                        for k1, v1 in config[k].items():
                            if not k1 in ['color_jitter', 'normalize']:
                                #key0 = list(v1.keys)[0]
                                cfg[f'{k1}_p'] = v1['p']
                            if (k1 == 'resize'):
                                if v1['p'] == None:
                                    v1['p'] == 0
                                else:
                                    if (v1['p']>0):
                                        cfg['img_size'] = v1['size']
                if k == 'criterion_kwargs':
                    for k2, v2 in config[k].items():
                        cfg['temperature'] = v2
                if k == 'model_kwargs':
                    for k3, v3 in config[k].items():
                        if k3 == 'features_dim':
                            cfg['proj_emb_size'] = v3
            hypp_dictionaries.append(cfg)
            if os.path.exists(os.path.join(target_parent, exp, 'latent_embeddings_train.pickle')) and os.path.exists(os.path.join(target_parent, exp, 'projected_embeddings_train.pickle')):
                trained.append(1)
            else:
                trained.append(0)
            if os.path.exists(os.path.join(target_parent, exp, 'plots')):
                results_computed.append(1)
            else:
                results_computed.append(0)

        summary_df = pd.DataFrame(hypp_dictionaries)
        summary_df['exp codes'] = all_exps
        summary_df['results_computed'] = results_computed
        summary_df['trained'] = trained
    summary_df.sort_values(by='exp codes', inplace=True)
    summary_df.to_csv(os.path.join(os.path.dirname(p['base_dir']), 'exp_summary.csv'), index=False)

    return summary_df

