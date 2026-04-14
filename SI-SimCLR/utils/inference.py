import torch
import numpy as np
import pdb
from utils.utils import compute_entropy, get_truncated_model
import os
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, Normalizer


def get_latent_embeddings(model, dataloader, base_directory, batch_size, set='train') -> np.ndarray:
    """
    Extract hidden features by inferring the truncated model.
    """

    if set != 'train' and set != 'val':
        raise ValueError('Specify which embeddings you want to retrieve (either train or val)')

    model = get_truncated_model(model)
    model.eval() # SET MODEL TO EVALUATION MODE
    device = next(model.parameters()).device
    indices = []
    latent_embeddings = []

    print('Starting to compute latent embeddings...')

    with torch.no_grad():
        for batch in dataloader: # From evaluate_utils (but modified)
            images = batch['image'].to(device, non_blocking=(device.type == 'cuda'))
            #target = batch['target'].cuda(non_blocking=True)
            output = model(images)
            for j in range(batch_size):
               if j < len(batch['meta']['index']):
                #print(int(batch['meta']['index'][j].numpy()))
                indices.append(int(batch['meta']['index'][j].numpy()))
                latent_embeddings.append(output[j])

    print('Computed latent embeddings...')
    latent_embeddings = np.asarray([latent_embeddings[i].cpu().detach().numpy() for i in range(len(latent_embeddings))])
    with open(os.path.join(base_directory, f'latent_embeddings_{set}.pickle'), 'wb') as handle:
        pickle.dump(latent_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #torch.save(latent_embeddings, os.path.join(base_directory, 'latent_embeddings_torch.pickle'))



def PCA_latent_embeddings(base_directory: str, val: bool):
    """
    This computes PCA and saves ALL 512 dimensions. To be reduced based on analysis.
    """
    latent_embeddings_TRAIN = pickle.load(open(os.path.join(base_directory, 'latent_embeddings_train.pickle'), 'rb'))
    if val:
        latent_embeddings_VAL = pickle.load(open(os.path.join(base_directory, 'latent_embeddings_val.pickle'), 'rb'))
    if not np.isnan(latent_embeddings_TRAIN).any():
        pca_ = PCA(.95, copy=False)
        scaler = StandardScaler().fit(latent_embeddings_TRAIN)
        scaled_embeddings_TR = scaler.transform(latent_embeddings_TRAIN)
        pca = pca_.fit(scaled_embeddings_TR)
        reduced_embeddings_TR = pca.transform(scaled_embeddings_TR)
        with open(os.path.join(base_directory, f'PCA_embeddings_train.pickle'), 'wb') as handle:
            pickle.dump(reduced_embeddings_TR, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Saved PCA for TRAINING set')
        # Initialize and build analisys dictionary
        print('Building analysis dictionary')
        analysis_dict = {}
        start_exp_code = base_directory.rfind('/')
        analysis_dict['exp_code'] = base_directory[start_exp_code+1:]
        analysis_dict['entropy'] = compute_entropy(latent_embeddings_TRAIN)
        analysis_dict['n_components_95_latent'] = pca.n_components_
        analysis_dict['variance_expl_20PCs_latent'] = np.sum(pca.explained_variance_ratio_[:20])*100
        # Save analysis dictionary
        with open(os.path.join(base_directory, 'analysis_dict.pickle'), 'wb') as handle:
            pickle.dump(analysis_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print('Latent embeddings contain NaN values. Skipping PCA computation...')

