import os
import ot
import pandas as pd
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

from joblib import Parallel, delayed
import networkx as nx
from scipy.stats import norm
from sklearn.mixture import GaussianMixture


# load un-processed partial OT results

hard_assignments = False
hard_assignments_str = '_hardassignment' if hard_assignments else ''
metric = 'euclidean'  # 'euclidean', 'sqeuclidean', 'cityblock'

transported_masses = pd.read_csv(f'./transported_masses_{metric}{hard_assignments_str}.csv', index_col=0)
transported_partial_distances = pd.read_csv(f'./transported_partial_distances_{metric}{hard_assignments_str}.csv', index_col=0)

transported_masses = np.array(transported_masses.values)
transported_partial_distances = np.array(transported_partial_distances.values)
print('transported_masses shape:', transported_masses.shape)
print('transported_partial_distances shape:', transported_partial_distances.shape)

# compute total partial costs
partial_OT_costs = transported_partial_distances.sum(axis=1)
print('partial_OT_costs:', partial_OT_costs)

scaling_strat = 'linear'

if scaling_strat == 'softmax':
    # compute normalized partial OT costs
    mean = partial_OT_costs.mean()
    print('mean:', mean)
    std = partial_OT_costs.std()
    print('std:', std)

    normalized_partial_OT_costs = (partial_OT_costs - mean) / std
    print('normalized_partial_OT_costs:', normalized_partial_OT_costs)

    # compute transition graph
    alphas = np.exp(-normalized_partial_OT_costs)
    alphas /= alphas.sum()
elif scaling_strat == 'linear':
    """
    “bounds were chosen to span the central dynamic range of transition propensities while limiting the influence of extreme clusters.”
    """
    # compute normalized partial OT costs
    min_cost = partial_OT_costs.min()
    print('min_cost:', min_cost)
    max_cost = partial_OT_costs.max()
    print('max_cost:', max_cost)
    lower_bound_cost = np.percentile(partial_OT_costs, 10)
    print('lower_bound_cost:', lower_bound_cost)
    upper_bound_cost = np.percentile(partial_OT_costs, 90)
    print('upper_bound_cost:', upper_bound_cost)
    clipped_costs = np.clip(partial_OT_costs, lower_bound_cost, upper_bound_cost)
    print('clipped_costs:', clipped_costs)
    alphas = (max_cost - clipped_costs) / (max_cost - min_cost)

print('alphas:', alphas)

n = len(alphas)
transition_matrix = np.zeros((n, n))

transition_matrix = alphas[:, None] * transported_masses
for i in range(n):
    transition_matrix[i, i] = 1 - alphas[i]


# Transition Matrix without Diagonal Components
transition_matrix_no_diag = transition_matrix.copy()
np.fill_diagonal(transition_matrix_no_diag, 0)

# Save the transition matrix to a CSV file
transition_matrix_df = pd.DataFrame(transition_matrix,  index=np.arange(1, n + 1), columns=np.arange(1, n + 1))
transition_matrix_df.to_csv(
    f'./transition_matrix_{metric}{hard_assignments_str}{scaling_strat}.csv',
   )
print(f"Transition matrix saved to ./transition_matrix_{metric}{hard_assignments_str}{scaling_strat}.csv")

# Visualize the transition matrix and its log-scale version jointly with transported masses and partial OT costs
plt.figure(figsize=(30, 6))

# Transported Masses
plt.subplot(1, 5, 1)
plt.imshow(transported_masses, cmap='viridis', aspect='auto')
plt.colorbar(label='Transported Mass')
plt.title('Transported Masses')
plt.xlabel('Target Cluster')
plt.ylabel('Source Cluster')

# Transported Partial Distances
plt.subplot(1, 5, 2)
plt.imshow(transported_partial_distances, cmap='viridis', aspect='auto')
plt.colorbar(label='Transported Partial Distance')
plt.title('Transported Partial Distances')
plt.xlabel('Target Cluster')
plt.ylabel('Source Cluster')

# Transition Matrix
plt.subplot(1, 5, 3)
plt.imshow(transition_matrix, cmap='viridis', aspect='auto')
plt.colorbar(label='Transition Probability')
plt.title('Transition Matrix')
plt.xlabel('Target Cluster')
plt.ylabel('Source Cluster')

# Transition Matrix without Diagonal Components
plt.subplot(1, 5, 4)
plt.imshow(transition_matrix_no_diag, cmap='viridis', aspect='auto')
plt.colorbar(label='Transition Probability (No Diagonal)')
plt.title('Transition Matrix (No Diagonal)')
plt.xlabel('Target Cluster')
plt.ylabel('Source Cluster')

# Log-Scale Transition Matrix
plt.subplot(1, 5, 5)
plt.imshow(np.log(transition_matrix + 1e-10), cmap='viridis', aspect='auto')  # Add small value to avoid log(0)
plt.colorbar(label='Log-Scale Transition Probability')
plt.title('Log-Scale Transition Matrix')
plt.xlabel('Target Cluster')
plt.ylabel('Source Cluster')

# Save the figure
plt.tight_layout()
plt.savefig(f'./transported_matrices_with_transition_no_diag_and_log_{metric}{hard_assignments_str}{scaling_strat}.pdf')
plt.show()

# Plot histogram of values in transition_matrix_no_diag (excluding diagonal zeros only)
non_diagonal_values = transition_matrix_no_diag[np.triu_indices_from(transition_matrix_no_diag, k=1)]
non_diagonal_values = np.append(non_diagonal_values, transition_matrix_no_diag[np.tril_indices_from(transition_matrix_no_diag, k=-1)])

plt.figure(figsize=(8, 6))
plt.hist(non_diagonal_values, bins=50, color='blue', alpha=0.7, edgecolor='black')
plt.title('Histogram of Values in Transition Matrix (Excluding Diagonal Zeros)')
plt.xlabel('Transition Probability')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f'./transition_matrix_no_diag_histogram_excluding_diagonal_{metric}{hard_assignments_str}{scaling_strat}.pdf')
plt.show()

# Fit a bi-modal Gaussian distribution to the histogram data
def fit_bimodal_gaussian(data):
    data = data.reshape(-1, 1)  # Reshape for GaussianMixture
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(data)

    means = gmm.means_.flatten()
    covariances = gmm.covariances_.flatten()
    weights = gmm.weights_.flatten()

    # Sort components by mean
    sorted_indices = np.argsort(means)
    means = means[sorted_indices]
    covariances = covariances[sorted_indices]
    weights = weights[sorted_indices]

    # Find threshold as midpoint between the two means
    threshold = (means[0] + means[1]) / 2

    return means, covariances, weights, threshold

# Fit the bi-modal Gaussian and plot the results
means, covariances, weights, threshold = fit_bimodal_gaussian(non_diagonal_values)

plt.figure(figsize=(8, 6))
plt.hist(non_diagonal_values, bins=50, density=True, color='blue', alpha=0.7, edgecolor='black', label='Histogram')

x = np.linspace(non_diagonal_values.min(), non_diagonal_values.max(), 1000)

# Plot the two Gaussian components
for mean, cov, weight in zip(means, covariances, weights):
    plt.plot(x, weight * norm.pdf(x, mean, np.sqrt(cov)), label=f'Gaussian (mean={mean:.2f})')

# Plot the threshold
plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.2f}')

plt.title('Bi-Modal Gaussian Fit and Threshold')
plt.xlabel('Transition Probability')
plt.ylabel('Density')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f'./bimodal_gaussian_fit_{metric}{hard_assignments_str}{scaling_strat}.pdf')
plt.show()

#%%% computation of graph metrics

# ------------------------------------------------------------
# 1. NET FLOW
# Δ_i = outgoing - incoming (excluding self-loops)
# ------------------------------------------------------------
P = transition_matrix_no_diag
K = P.shape[0]

out_flow = P.sum(axis=1)
in_flow = P.sum(axis=0)
net_flow = out_flow - in_flow


# ------------------------------------------------------------
# 2. GENERALISED ENTROPY (as in your LaTeX)
# H_gen(i) = - Σ w_ij log(w_ij / u_j) - w_ij + u_j
# with u_j = 1/(K-1)
# ------------------------------------------------------------

def generalized_entropy(P):
    K = P.shape[0]
    eps = 1e-15  # numerical stability

    u = 1.0 / (K - 1)

    H = np.zeros(K)

    for i in range(K):
        w = P[i, :]
        # exclude self-loop already zeroed
        w = np.clip(w, eps, None)

        term = (
            w * np.log(w / u)
            - w
            + u
        )

        H[i] = -np.sum(term)

    return H

gen_entropy_out = generalized_entropy(P)
gen_entropy_in = generalized_entropy(P.T)



# ------------------------------------------------------------
# 3. BETWENNESS CENTRALITY (NetworkX)
# edge weight = cost = 1 / P_ij
# ------------------------------------------------------------

sp_G = nx.DiGraph()

for i in range(K):
    for j in range(K):
        if P[i, j] > 0:
            cost = 1.0 / P[i, j]
            sp_G.add_edge(i, j, weight=cost)

betweenness = nx.betweenness_centrality(
    sp_G,
    weight="weight",
    normalized=True
)


# convert dict → array
betweenness = np.array([betweenness[i] for i in range(K)])


# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

results = {
    "net_flow": net_flow,
    "generalized_entropy_out": gen_entropy_out,
    "generalized_entropy_in": gen_entropy_in,
    "betweenness": betweenness
}

print(results)
# %%

net_flow = results["net_flow"]
betweenness = results["betweenness"]

# ------------------------------------------------------------
# GRAPH
# ------------------------------------------------------------
G = nx.DiGraph()

for i in range(K):
    for j in range(K):
        if P[i, j] > 0:
            G.add_edge(i + 1, j + 1, weight=P[i, j])

pos = nx.spring_layout(G, seed=42)

nodes = np.arange(1, K + 1)


# ------------------------------------------------------------
# helper function: draw graph with node coloring
# ------------------------------------------------------------
def draw_metric(ax, values, title, cmap="viridis"):
    ax.set_title(title)

    values = np.array(values)

    norm = Normalize(vmin=np.min(values), vmax=np.max(values))
    colors = cm.get_cmap(cmap)(norm(values))

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=colors,
        node_size=600,
        ax=ax
    )

    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9)

    edges = G.edges(data=True)
    weights = [d["weight"] for (_, _, d) in edges]
    probabilities = [d.get("probability", 1.0) for (_, _, d) in edges]  # Default to 1.0 if not provided

    for (u, v, d), prob in zip(edges, probabilities):
        edge_color = str(1 - prob)  # Convert probability to grayscale intensity
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            width=d["weight"] * 3.0,
            alpha=0.8,
            edge_color=edge_color,
            ax=ax,
            connectionstyle="arc3,rad=0.2"  # Add curvature to edges
        )

    ax.axis("off")

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)


# ------------------------------------------------------------
# FIGURE
# ------------------------------------------------------------
fig = plt.figure(figsize=(14, 10))

ax1 = plt.subplot(2, 2, 1)
draw_metric(ax1, net_flow, "Net Flow")

ax2 = plt.subplot(2, 2, 2)
draw_metric(ax2, gen_entropy_out, "Outgoing Entropy")

ax3 = plt.subplot(2, 2, 3)
draw_metric(ax3, gen_entropy_in, "Incoming Entropy")

ax4 = plt.subplot(2, 2, 4)
draw_metric(ax4, betweenness, "Betweenness Centrality")

plt.tight_layout()
plt.savefig(f'./graph_metrics_{metric}{hard_assignments_str}{scaling_strat}.pdf')
plt.show()


# Save results as a CSV file
def save_results_to_csv(metrics, cluster_numbers, output_file="node_metric_results.csv"):
    """
    Save metrics to a CSV file with columns as metric names and rows as cluster numbers.

    Parameters:
    - metrics: dict, where keys are metric names and values are lists of metric values.
    - cluster_numbers: list, cluster numbers corresponding to rows.
    - output_file: str, path to save the CSV file.
    """
    df = pd.DataFrame(metrics, index=cluster_numbers)
    df.index.name = "Cluster Number"
    df.to_csv(output_file)
    print(f"Results saved to {output_file}")

# Example usage
metrics = {
    "Partial OT Costs": partial_OT_costs,
    "Alphas": alphas,
    "Net Flow": net_flow,
    "Generalized Entropy Out": gen_entropy_out,
    "Generalized Entropy In": gen_entropy_in,
    "Betweenness Centrality": betweenness
}
cluster_numbers = list(range(1, len(partial_OT_costs) + 1)) 
save_results_to_csv(metrics, cluster_numbers)


# ------------------------------------------------------------
# Pruned GRAPH visualization
# ------------------------------------------------------------
G_pruned = nx.DiGraph()

for i in range(K):
    for j in range(K):
        if P[i, j] > threshold:
            G_pruned.add_edge(i + 1, j + 1, weight=P[i, j])

pos_pruned = nx.spring_layout(G_pruned, seed=42)

nodes = np.arange(1, K + 1)


# ------------------------------------------------------------
# helper function: draw graph with node coloring
# ------------------------------------------------------------
def draw_metric(ax, values, title, cmap="viridis"):
    ax.set_title(title)

    values = np.array(values)

    norm = Normalize(vmin=np.min(values), vmax=np.max(values))
    colors = cm.get_cmap(cmap)(norm(values))

    nx.draw_networkx_nodes(
        G_pruned,
        pos_pruned,
        node_color=colors,
        node_size=600,
        ax=ax
    )

    nx.draw_networkx_labels(G_pruned, pos_pruned, ax=ax, font_size=9)

    edges = G_pruned.edges(data=True)
    weights = [d["weight"] for (_, _, d) in edges]
    probabilities = [d.get("probability", 1.0) for (_, _, d) in edges]  # Default to 1.0 if not provided

    for (u, v, d), prob in zip(edges, probabilities):
        edge_color = str(1 - prob)  # Convert probability to grayscale intensity
        nx.draw_networkx_edges(
            G_pruned,
            pos_pruned,
            edgelist=[(u, v)],
            width=d["weight"] * 3.0,
            alpha=0.8,
            edge_color=edge_color,
            ax=ax,
            connectionstyle="arc3,rad=0.2"  # Add curvature to edges
        )

    ax.axis("off")

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)


# ------------------------------------------------------------
# FIGURE
# ------------------------------------------------------------
fig = plt.figure(figsize=(14, 10))

ax1 = plt.subplot(2, 2, 1)
draw_metric(ax1, net_flow, "Net Flow")

ax2 = plt.subplot(2, 2, 2)
draw_metric(ax2, gen_entropy_out, "Outgoing Entropy")

ax3 = plt.subplot(2, 2, 3)
draw_metric(ax3, gen_entropy_in, "Incoming Entropy")

ax4 = plt.subplot(2, 2, 4)
draw_metric(ax4, betweenness, "Betweenness Centrality")

plt.tight_layout()
plt.savefig(f'./pruned_graph_metrics_{metric}{hard_assignments_str}{scaling_strat}.pdf')
plt.show()
