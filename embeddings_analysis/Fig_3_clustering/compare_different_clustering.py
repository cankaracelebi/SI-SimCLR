import sys
from pathlib import Path

# Ensure project-root imports (e.g., utils.*) work regardless launch directory.
SCRIPT_DIR = Path(__file__).resolve().parent
for candidate in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
    if (candidate / 'utils').is_dir():
        if str(candidate) not in sys.path:
            sys.path.append(str(candidate))
        break

import numpy as np
import pandas as pd
import pickle
import os
import math
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['legend.markerscale'] = 1.5
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from sklearn.metrics import pairwise_distances, silhouette_score, v_measure_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.cluster import KMeans, AgglomerativeClustering, HDBSCAN
from sklearn.neighbors import NearestNeighbors
import umap
from utils.mypath import MyPath
from utils.config import load_env_paths
#from utils.clustering_analysis import compute_cluster_entropies
from utils.snn import *
from utils.start_notebook import setup_experiment, load_process_handcrafted_features
from sklearnex import patch_sklearn
patch_sklearn()

# Set random seed for reproducibility
np.random.seed(0)
  
warnings.filterwarnings('ignore')

single_exps = ['E1']
emb_type = 'deep'
assert emb_type in ['handcrafted', 'deep']

x = '240813_5'
label_of_interest = 'mut_cond'

# Resolve output roots from repo config.
REPO_ROOT = next(candidate for candidate in [SCRIPT_DIR, *SCRIPT_DIR.parents] if (candidate / 'configs' / 'env.yml').is_file())
ENV_PATHS = load_env_paths(str(REPO_ROOT / 'configs' / 'env.yml'))

saving_dir = os.path.join(ENV_PATHS['temp_results_root_dir'], x, 'clustering', 'UMAP')

if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)

name_file = os.path.join(saving_dir, f'clustering_algo_comparison_{("_").join(single_exps)}.csv')

print('Results will be saved as: ', name_file, flush=True)

annotations_df, raw_emb = setup_experiment(x, single_exps)
if emb_type == 'deep':
    emb = raw_emb
else:
    hand_crafted_annotations, hc_features = load_process_handcrafted_features(annotations_df)
    emb = hc_features


embeddings = {}
# Compute UMAP embeddings with a range of n_neighbors parameter
#n_neighbors_range = [15, 50, 100, 150, 200, 300, 600]
n_neighbors_range = [15, 50]
for n_neighbors in n_neighbors_range:
    umap_emb = umap.UMAP(n_neighbors=n_neighbors, random_state=0).fit_transform(emb)
    embeddings[f'umap_n_neighbors_{n_neighbors}'] = umap_emb

clustering_hypp = {
    #'KMeans': {'n_clusters': range(2, 21), 'random_state': 0, },
    #'HDBSCAN': {'min_cluster_size': range(20, 110, 10)},
    #'SNN': {'neighbor_num': range(50, 220, 25), 'min_shared_neighbor_proportion': [0.6, 0.7, 0.8, 0.9]}
    'KMeans': {'n_clusters': range(2, 3), 'random_state': 0, },
    'HDBSCAN': {'min_cluster_size': range(80, 110, 20)},
    'SNN': {'neighbor_num': range(50, 220, 70), 'min_shared_neighbor_proportion': [0.7, 0.9]}
}

# In case you already have results for some clustering algorithms, you might want to just update the results file.

#if os.path.exists(os.path.join('YOUR/PATH/DL_analysis/temp_results', f'clustering_results_{single_exps[0]}.csv')):
#    previous_clustering_results = pd.read_csv(os.path.join('YOUR/PATH/DL_analysis/temp_results', f'clustering_results_{single_exps[0]}.csv'))
#    if 'neighbor_num' not in previous_clustering_results.columns:
#        previous_clustering_results['neighbor_num'] = np.nan
#    if 'min_shared_neighbor_proportion' not in previous_clustering_results.columns:
#        previous_clustering_results['min_shared_neighbor_proportion'] = np.nan
#    for method in clustering_hypp.keys():
#        for param in clustering_hypp[method].keys():
#            if not(param in (['random_state', 'linkage'])):
#                previous_params = previous_clustering_results[previous_clustering_results['method'] == method][param].unique()
#                clustering_hypp[method][param] = [p for p in clustering_hypp[method][param] if p not in previous_params]


print(clustering_hypp, flush=True)

results = []

KNN_K = 25
UNASSIGNED_THRESHOLD = 0.15


def apply_knn_to_unassigned(emb, labels, k=KNN_K):
    labels = np.asarray(labels).copy()
    unassigned_mask = labels == -1
    if not np.any(unassigned_mask):
        return labels

    assigned_mask = ~unassigned_mask
    if not np.any(assigned_mask):
        return labels

    k_eff = min(k, int(np.sum(assigned_mask)))
    nn = NearestNeighbors(n_neighbors=k_eff)
    nn.fit(emb[assigned_mask])

    _, neighbor_idx = nn.kneighbors(emb[unassigned_mask])
    assigned_labels = labels[assigned_mask]

    for row_idx, neigh in enumerate(neighbor_idx):
        neigh_labels = assigned_labels[neigh]
        unique_labels, counts = np.unique(neigh_labels, return_counts=True)
        labels[np.where(unassigned_mask)[0][row_idx]] = unique_labels[np.argmax(counts)]

    return labels

def evaluate_clustering(emb_name, emb, labels, clustering_method, **kwargs):
    labels = np.asarray(labels)
    total_points = len(labels)
    unassigned = int(np.sum(labels == -1))
    unassigned_fraction = unassigned / total_points
    knn_applied = False

    labels_for_metrics = labels.copy()
    if 0 < unassigned_fraction < UNASSIGNED_THRESHOLD:
        labels_for_metrics = apply_knn_to_unassigned(emb, labels_for_metrics, k=KNN_K)
        knn_applied = True

    # Exclude only runs with too many unassigned points.
    include_for_plots = (unassigned_fraction <= UNASSIGNED_THRESHOLD)

    valid_labels = labels_for_metrics != -1
    filtered_emb = emb[valid_labels]
    filtered_labels = labels_for_metrics[valid_labels]
    filtered_annotations_df = annotations_df[valid_labels]

    unique_clusters = np.unique(filtered_labels)

    if len(unique_clusters) <= 1:
        return {
            'embedding': emb_name,
            'method': clustering_method,
            'ARI': 0,
            'vmeasure': 0,
            'silhouette': 0,
            'calinski_harabasz': 0,
            'unassigned': unassigned,
            'unassigned_fraction': unassigned_fraction,
            'knn_applied': knn_applied,
            'include_for_plots': include_for_plots,
            'n_clusters': len(unique_clusters),
            'labels': labels,
            'labels_for_metrics': labels_for_metrics,
            **kwargs
        }
    else:
        return {
            'embedding': emb_name,
            'method': clustering_method,
            'ARI': adjusted_rand_score(filtered_annotations_df[label_of_interest], filtered_labels),
            'vmeasure': v_measure_score(filtered_annotations_df[label_of_interest], filtered_labels),
            'silhouette': silhouette_score(filtered_emb, filtered_labels),
            'calinski_harabasz': calinski_harabasz_score(filtered_emb, filtered_labels),
            'unassigned': unassigned,
            'unassigned_fraction': unassigned_fraction,
            'knn_applied': knn_applied,
            'include_for_plots': include_for_plots,
            'n_clusters': len(unique_clusters),
            'labels': labels,
            'labels_for_metrics': labels_for_metrics,
            **kwargs
        }

def run_kmeans(emb_name, emb, hypp):
    results = []
    print('Running KMeans')
    for n_clusters in hypp['n_clusters']:
        model = KMeans(n_clusters=n_clusters, random_state=hypp['random_state'])
        labels = model.fit_predict(emb)
        results.append(evaluate_clustering(emb_name, emb, labels, 'KMeans', n_clusters=n_clusters))
    return results

def run_agglomerative(emb_name, emb, hypp):
    results = []
    print('Running Agglomerative')
    for n_clusters in hypp['n_clusters']:
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=hypp['linkage'])
        labels = model.fit_predict(emb)
        results.append(evaluate_clustering(emb_name, emb, labels, 'Agglomerative', n_clusters=n_clusters))
    return results

def run_hdbscan(emb_name, emb, hypp):
    results = []
    print('Running HDBSCAN')
    for min_cluster_size in hypp['min_cluster_size']:
        model = HDBSCAN(min_cluster_size=min_cluster_size)
        labels = model.fit_predict(emb)
        results.append(evaluate_clustering(emb_name, emb, labels, 'HDBSCAN', min_cluster_size=min_cluster_size))
    return results

def run_snn(emb_name, emb, hypp):
    results = []
    print('Running SNN')
    for neighbor_num in hypp['neighbor_num']:
        for min_shared_neighbor_proportion in hypp['min_shared_neighbor_proportion']:
            model = SNN(neighbor_num=neighbor_num, min_shared_neighbor_proportion=min_shared_neighbor_proportion, clustering_algo='DBSCAN')
            labels = model.fit_predict(emb)
            results.append(evaluate_clustering(emb_name, emb, labels, 'SNN', neighbor_num=neighbor_num, min_shared_neighbor_proportion=min_shared_neighbor_proportion))
    return results

def run_clustering(embeddings, clustering_hypp):
    results = []
    for emb_name, emb in tqdm(embeddings.items()):
        for clustering_method, hypp in clustering_hypp.items():
            if clustering_method == 'KMeans':
                results.extend(run_kmeans(emb_name, emb, hypp))
            elif clustering_method == 'Agglomerative':
                results.extend(run_agglomerative(emb_name, emb, hypp))
            elif clustering_method == 'HDBSCAN':
                results.extend(run_hdbscan(emb_name, emb, hypp))
            elif clustering_method == 'SNN':
                #continue
                results.extend(run_snn(emb_name, emb, hypp))
    return pd.DataFrame(results)


def save_metric_plots(clustering_results, saving_dir):
    plot_df = clustering_results[clustering_results['include_for_plots']].copy()

    if plot_df.empty:
        print('No eligible configurations for plotting (all runs had unassigned points).', flush=True)
        return

    def _plot_method(df, method_name, out_name):
        method_df = df[df['method'] == method_name]
        if method_df.empty:
            print(f'No {method_name} results to plot.', flush=True)
            return

        kwargs = {
            'hue': 'embedding',
            'data': method_df,
            'x': 'n_clusters',
            'marker': 'o',
            'alpha': 0.8,
            'markersize': 5,
            'errorbar': None,
        }

        fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True)
        sns.lineplot(y='silhouette', ax=ax[0], **kwargs)
        ax[0].set_title(f'Silhouette for {method_name}')
        ax[0].set_ylabel('Silhouette')

        sns.lineplot(y='ARI', ax=ax[1], **kwargs)
        ax[1].set_title(f'ARI for {method_name}')
        ax[1].set_ylabel('ARI')

        handles, labels = ax[0].get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(1.00, 0.95), loc='upper left')

        for ax_ in ax.flatten():
            if ax_.legend_ is not None:
                ax_.legend_.remove()
            ax_.grid(axis='x')

        plt.tight_layout()
        plt.savefig(os.path.join(saving_dir, f'{out_name}.pdf'), format='PDF')
        plt.savefig(os.path.join(saving_dir, f'{out_name}.png'), format='PNG', dpi=300)
        plt.close(fig)

    _plot_method(plot_df, 'KMeans', 'KMeans_silhouette_ARI')
    _plot_method(plot_df, 'SNN', 'SNN_silhouette_ARI')

clustering_results = run_clustering(embeddings, clustering_hypp)

clustering_results.to_csv(name_file, index=False)
print('Results have been saved')

save_metric_plots(clustering_results, saving_dir)
print('Silhouette and ARI plots have been saved', flush=True)
