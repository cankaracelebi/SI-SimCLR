import os
import ot
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

data_root = "../../embeddings/240813_5/"
emb_path = data_root + "E1_emb.feather"

annotations_df = pd.read_feather(emb_path)
emb_cols = [col for col in annotations_df.columns if col.startswith('emb')]

print('columns:', annotations_df.columns)
print('shape:', annotations_df.shape)

u_cluster, c_cluster = np.unique(np.array(annotations_df['cluster'].values), return_counts=True)
print('Cluster labels:', u_cluster, c_cluster)

embeddings = np.array(annotations_df[emb_cols].values).astype(np.float32)
clusters = np.array(annotations_df['cluster'].values)
n_clusters = len(u_cluster)

metric = 'euclidean'  # 'euclidean', 'sqeuclidean', 'cityblock'

print('embeddings shape:', embeddings.shape)
if metric in ['euclidean', 'sqeuclidean']:
    D = ot.dist(embeddings, embeddings, metric=metric)
elif metric == 'cityblock':
    # long computation to parallelize with joblib
    D = np.zeros((embeddings.shape[0], embeddings.shape[0]))
    n_jobs = -1
    def compute_distance(i):
        return ot.dist(embeddings[i, :].reshape(1, -1), embeddings[i + 1:, :], metric=metric).flatten()
    if n_jobs is None:
        results = []
        for i in tqdm(range(embeddings.shape[0] - 1)):
            dists = compute_distance(i)
            results.append(dists)
            
    else:
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(compute_distance)(i) for i in range(embeddings.shape[0] - 1))
    
    # fill the distance matrix
    for i in range(embeddings.shape[0] - 1):
        D[i, i + 1:] = results[i]
    D = D + D.T  # make it symmetric
    
hard_assignments = True
hard_assignments_str = '_hardassignment' if hard_assignments else ''

if not hard_assignments:
    masses = np.zeros(embeddings.shape[0])
    # assign total mass of one to each cluster independently
    for i in range(len(u_cluster)):
        masses[clusters == u_cluster[i]] = 1.0 / c_cluster[i]
    
def build_onevsall_distributions(id):
    # get the indices corresponding to the cluster
    source_indices = np.argwhere(clusters == id)[:, 0]
    target_indices = np.argwhere(clusters != id)[:, 0]
    print('source_indices:', source_indices.shape)
    print('target_indices:', target_indices.shape)
    # get the pairwise distances
    D_sub = D[source_indices][:, target_indices]    
    print('D_sub:', D_sub.shape)
    # normalize masses per cluster  
    if not hard_assignments:
        source_masses = masses[source_indices].flatten()
        target_masses = masses[target_indices].flatten()
    else:
        source_masses = np.ones(len(source_indices)) / len(source_indices)
        target_masses = np.ones(len(target_indices)) / len(source_indices)
        
    return D_sub, source_masses, target_masses, target_indices
    
# Compute one versus all partial OT masses and distances
transported_masses = np.zeros((n_clusters, n_clusters))
transported_partial_distances = np.zeros((n_clusters, n_clusters))
transported_avg_distances = np.zeros((n_clusters, n_clusters))

for i in tqdm(range(n_clusters)):
    D_sub, source_masses, target_masses, target_indices = build_onevsall_distributions(u_cluster[i])
    # compute the optimal transport plan
    m = np.sum(source_masses)
    assert np.isclose(m, 1.0), f"Source masses for cluster {u_cluster[i]} do not sum to 1."
    T, log = ot.partial.partial_wasserstein(
        source_masses, target_masses, D_sub,
        m=m,
        numItermax=1e5, log=True)
    total_cost = log['partial_w_dist']
    print('total_mass: ', T.sum())
    print('variation source marginal:', np.abs(np.sum(T, axis=1) - source_masses).sum())
    print('variation target marginal:', np.abs(np.sum(T, axis=0) - target_masses).sum())

    # identify per target cluster the transported mass and distance
    for j in range(n_clusters):
        if j == i:
            continue  # skip the same cluster
        target_cluster_indices = np.argwhere(clusters[target_indices] == u_cluster[j])[:, 0]
        if len(target_cluster_indices) == 0:
            continue  # skip if no target cluster samples
        transported_mass = T[:, target_cluster_indices].sum()
        transported_distance = (T[:, target_cluster_indices] * D_sub[:, target_cluster_indices]).sum()
        transported_masses[i, j] = transported_mass
        transported_partial_distances[i, j] = transported_distance


# Add a third plot for the sparse map
plt.figure(figsize=(12, 6))

# Sparse Map
sparse_map = (transported_masses == 0) 
sparsity_level = np.round((sparse_map.sum() - n_clusters) / (n_clusters * (n_clusters-1)) * 100, 2)
print('Sparsity level: {:.2f}%'.format(sparsity_level))

plt.imshow(sparse_map, cmap='gray', aspect='auto')
plt.colorbar(label='Sparse Map (1 = Zero)')
plt.title(f'Sparse Map of Zero Values ({sparsity_level} % sparse)')
plt.xlabel('Target Cluster')
plt.ylabel('Source Cluster')
plt.xticks(ticks=np.arange(n_clusters), labels=u_cluster, rotation=90)
plt.yticks(ticks=np.arange(n_clusters), labels=u_cluster)

# Save the sparse map figure
plt.tight_layout()
plt.savefig(f'./sparse_map_{metric}{hard_assignments_str}.pdf')

plt.show()


# visualize via imshow and save results
plt.figure(figsize=(18, 6))

# Transported Masses
plt.subplot(1, 3, 1)
plt.imshow(transported_masses, cmap='viridis', aspect='auto')
plt.colorbar(label='Transported Mass')
plt.title('Transported Masses')
plt.xlabel('Target Cluster')
plt.ylabel('Source Cluster')
plt.xticks(ticks=np.arange(n_clusters), labels=u_cluster, rotation=90)
plt.yticks(ticks=np.arange(n_clusters), labels=u_cluster)

# Transported Partial Distances
plt.subplot(1, 3, 2)
plt.imshow(transported_partial_distances, cmap='viridis', aspect='auto')
plt.colorbar(label='Transported Partial Distance')
plt.title('Transported Partial Distances')
plt.xlabel('Target Cluster')
plt.ylabel('Source Cluster')
plt.xticks(ticks=np.arange(n_clusters), labels=u_cluster, rotation=90)
plt.yticks(ticks=np.arange(n_clusters), labels=u_cluster)

# Sparse Map
plt.subplot(1, 3, 3)
sparse_map = (transported_masses == 0) & (transported_partial_distances == 0)
plt.imshow(sparse_map, cmap='gray', aspect='auto')
plt.colorbar(label='Sparse Map (1 = Zero)')
plt.title(f'Sparse Map of Zero Values ({sparsity_level} % sparse)')

plt.xlabel('Target Cluster')
plt.ylabel('Source Cluster')
plt.xticks(ticks=np.arange(n_clusters), labels=u_cluster, rotation=90)
plt.yticks(ticks=np.arange(n_clusters), labels=u_cluster)

# Save the figure
plt.tight_layout()
plt.savefig(f'./transported_matrices_comparison_with_sparse_map_{metric}{hard_assignments_str}.pdf')

plt.show()

# save the transported masses and distances as csv files

pd.DataFrame(transported_masses, index=u_cluster, columns=u_cluster).to_csv(
    f'./transported_masses_{metric}{hard_assignments_str}.csv')
pd.DataFrame(transported_partial_distances, index=u_cluster, columns=u_cluster).to_csv(
    f'./transported_partial_distances_{metric}{hard_assignments_str}.csv')