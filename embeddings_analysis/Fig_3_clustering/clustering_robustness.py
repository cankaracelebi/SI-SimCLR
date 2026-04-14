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
import pickle
import os
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['legend.markerscale'] = 1.5
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['savefig.dpi'] = 300
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import argparse
import umap
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans, AgglomerativeClustering, HDBSCAN
from sklearn.neighbors import NearestNeighbors
from utils.snn import SNN
from utils.config import load_env_paths
from utils.start_notebook import setup_experiment
from sklearnex import patch_sklearn
patch_sklearn()
  
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Clustering robustness analysis')
parser.add_argument('--n_runs', type=int, default=50, help='Number of clustering runs')
parser.add_argument('--clustering', type=str, default='kmeans', choices=['kmeans', 'agglomerative', 'hdbscan', 'SNN'],
                    help='Clustering algorithm to use')
parser.add_argument('--neighbor_num', type=int, default=50, help='Number of neighbors (for SNN)')
parser.add_argument('--min_shared_proportion', type=float, default=0.5, help='Minimum shared proportion (for SNN)')
parser.add_argument('--n_clusters', type=int, default=6, help='Number of clusters (for kmeans/agglomerative)')
parser.add_argument('--hdbscan_min_cluster_size', type=int, default=10, help='min_cluster_size for HDBSCAN')
parser.add_argument('--hdbscan_min_samples', type=int, default=1, help='min_samples for HDBSCAN')
parser.add_argument('--umap_min_neighbors', type=int, default=15, help='min_neighbors for UMAP')
args = parser.parse_args()

clustering_name = args.clustering
clustering_params = {}

if clustering_name == 'kmeans':
    clustering_params['n_clusters'] = args.n_clusters
    clustering = KMeans(**clustering_params)
elif clustering_name == 'SNN':
    clustering_params['neighbor_num'] = args.neighbor_num
    clustering_params['min_shared_neighbor_proportion'] = args.min_shared_proportion
    clustering = SNN(**clustering_params)
elif clustering_name == 'agglomerative':
    clustering_params['n_clusters'] = args.n_clusters
    clustering_params['linkage'] = 'ward'
    clustering = AgglomerativeClustering(**clustering_params)
elif clustering_name == 'hdbscan':
    clustering_params['min_cluster_size'] = args.hdbscan_min_cluster_size
    clustering_params['min_samples'] = args.hdbscan_min_samples
    clustering = HDBSCAN(**clustering_params)

N_RUNS = args.n_runs
random_seeds = np.arange(N_RUNS)

x = '240813_5'
single_exp = ['E4']
REPO_ROOT = next(candidate for candidate in [SCRIPT_DIR, *SCRIPT_DIR.parents] if (candidate / 'configs' / 'env.yml').is_file())
ENV_PATHS = load_env_paths(str(REPO_ROOT / 'configs' / 'env.yml'))
saving_dir = os.path.join(ENV_PATHS['temp_results_root_dir'], x, 'clustering_robustness', ('_').join(single_exp))
os.makedirs(saving_dir, exist_ok=True)


annotations_df, raw_emb = setup_experiment(x, single_exp)
labels_runs = np.zeros((N_RUNS, raw_emb.shape[0]), dtype=int)
umap_n_neighbors = args.umap_min_neighbors

# 1) Run clustering multiple times
for i, seed in enumerate(tqdm(random_seeds, desc="Clustering runs")):
    umap_emb = umap.UMAP(n_neighbors=umap_n_neighbors, random_state=int(seed)).fit_transform(raw_emb)
    labels = clustering.fit_predict(umap_emb)

    # Assign unassigned points (-1) to nearest cluster
    unassigned = np.where(labels == -1)[0]
    assigned = np.where(labels != -1)[0]
    if unassigned.size > 0 and assigned.size > 0:
        nn = NearestNeighbors(n_neighbors=25).fit(umap_emb[assigned])
        dists, idxs = nn.kneighbors(umap_emb[unassigned])
        labels[unassigned] = labels[assigned][idxs[:, 0]]

    labels_runs[i] = labels

print("Finished clustering multiple runs.", flush=True)
# Save labels of multiple runs
labels_path = os.path.join(saving_dir, f"clustering_labels_{clustering_name}_{('_').join(single_exp)}.npy")
np.save(labels_path, labels_runs)
print(f"Clustering labels saved to {labels_path}", flush=True)

# 2) Pairwise agreement between runs
ari_mat = np.zeros((N_RUNS, N_RUNS))
for i in range(N_RUNS):
    for j in range(i+1, N_RUNS):
        a = adjusted_rand_score(labels_runs[i], labels_runs[j])
        ari_mat[i,j] = ari_mat[j,i] = a

ari_values = ari_mat[np.triu_indices(N_RUNS, k=1)]
print("ARI mean, std:", ari_values.mean(), ari_values.std(), flush=True)


# 3) Co-association (consensus) matrix
n_samples = raw_emb.shape[0]
coassoc = np.zeros((n_samples, n_samples), dtype=float)
for run in range(N_RUNS):
    lab = labels_runs[run]
    for c in np.unique(lab):
        idx = np.where(lab == c)[0]
        if idx.size > 1:
            coassoc[np.ix_(idx, idx)] += 1

coassoc = coassoc / float(N_RUNS)  # values in [0,1]

coassoc_path = os.path.join(saving_dir, f"coassociation_matrix_{('_').join(single_exp)}.npy")
np.save(coassoc_path, coassoc)
print(f"Co-association matrix saved to {coassoc_path}", flush=True)


# 3) Consensus clustering from co-association
dist = 1.0 - coassoc
np.fill_diagonal(dist, 0.0)
condensed = squareform(dist, checks=False)
Z = linkage(condensed, method='average')


# Use the requested number of consensus clusters.
final_K = max(2, min(args.n_clusters, n_samples))
consensus_labels = fcluster(Z, t=final_K, criterion='maxclust') - 1

consensus_labels_path = os.path.join(
    saving_dir,
    f"consensus_labels_{clustering_name}_{('_').join(single_exp)}.npy",
)
np.save(consensus_labels_path, consensus_labels)
print(f"Consensus labels saved to {consensus_labels_path}", flush=True)

# 7) Co-association heatmap (hierarchical ordering)
order = leaves_list(Z)
coassoc_ordered = coassoc[np.ix_(order, order)]

fig, ax = plt.subplots(figsize=(10, 8))
hm = sns.heatmap(
    coassoc_ordered,
    cmap="viridis",
    square=True,
    xticklabels=False,
    yticklabels=False,
    cbar_kws={'label': 'Co-association Probability'},
    ax=ax,
)
for collection in hm.collections:
    collection.set_rasterized(True)

ax.set_title(f"Co-association Matrix (N={N_RUNS} runs)")
ax.set_xlabel("Samples (ordered by consensus hierarchy)")
ax.set_ylabel("Samples (ordered by consensus hierarchy)")

heatmap_path = os.path.join(saving_dir, "coassociation_heatmap_rasterized.pdf")
fig.savefig(heatmap_path, bbox_inches='tight')
plt.close(fig)
print(f"Heatmap saved to: {heatmap_path}", flush=True)



# Print and save summary of results ----------
summary = {
    "ARI_mean": float(ari_values.mean()),
    "ARI_std": float(ari_values.std()),
    "N_runs": int(N_RUNS),
    "Clustering": clustering_name,
    "Clustering_params": clustering_params,
    "Consensus_n_clusters": int(final_K),
    "exp_id": x,
    "single_exp": single_exp,
    "labels_path": labels_path,
    "coassociation_matrix_path": coassoc_path,
    "consensus_labels_path": consensus_labels_path,
    "heatmap_path": heatmap_path,
}

print("\n===== SUMMARY =====")
for k, v in summary.items():
    print(f"{k}: {v}")

summary_path = os.path.join(saving_dir, f"clustering_robustness_summary_{('_').join(single_exp)}.pkl")
with open(summary_path, "wb") as f:
    pickle.dump(summary, f)
print(f"\nSummary saved to {summary_path}")