import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, Wedge
from matplotlib.colors import to_hex
from scipy.sparse import csgraph, linalg, find
import igraph as ig
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import pairwise_distances


def build_graph_and_layout(matrix, adjacency=True, edge_threshold=0.00, layout_type="fr"):
    from scipy.sparse import csr_matrix

    if adjacency:
        row_sums = np.array(matrix.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1
        matrix = matrix*(1 / row_sums[:, np.newaxis])
    matrix = csr_matrix(matrix)

    print(f"The matrix has {matrix.nnz} non-zero elements.")

    # Set elements of cluster_adjacency below the threshold to 0
    matrix.data[matrix.data < edge_threshold] = 0
    matrix.eliminate_zeros()

    print(f"After thresholding, the matrix has {matrix.nnz} non-zero elements.")

    rows, cols, weights = find(matrix)
    graph = ig.Graph(directed=True)
    graph.add_vertices(matrix.shape[0])
    graph.add_edges(zip(rows, cols))
    graph.es['weight'] = weights

    layout = np.array(graph.layout(layout_type))

    return graph, layout


def plot_graph_with_annotations(graph, layout, annotations, color_by='cluster', terminal_clusters=None, palette=None, node_size=300, highlighted_edges=None, title=None):

    pie_data = {}
    vertex_colors = []

    if min(annotations['cluster']) > 0:
        cluster_0 = False
    else:
        cluster_0 = True

    if color_by == 'pseudotime':
        pt_norm = (annotations['pseudotime'] - annotations['pseudotime'].min()) / \
                  (annotations['pseudotime'].max() - annotations['pseudotime'].min())
        cluster_means = [pt_norm[annotations['cluster'] == c].mean() for c in range(graph.vcount())]
        cmap = plt.get_cmap(palette if palette else 'viridis')
        vertex_colors = [to_hex(cmap(val)) for val in cluster_means]

    elif color_by in annotations.columns:
        if isinstance(palette, str):
            unique_vals = annotations[color_by].unique()
            palette = dict(zip(unique_vals, sns.color_palette(palette, len(unique_vals))))
        elif palette is None:
            unique_vals = annotations[color_by].unique()
            palette = dict(zip(unique_vals, sns.color_palette("tab20", len(unique_vals))))

        for i in range(graph.vcount()):
            annot = annotations[annotations['cluster'] == i+(not cluster_0)]
            value_counts = annot[color_by].value_counts(normalize=True)
            value_counts = value_counts[value_counts > 0]
            if len(value_counts) == 1:
                vertex_colors.append(to_hex(palette[value_counts.index[0]-(not cluster_0)]))
            else:
                pie_data[i] = value_counts.to_dict()
                vertex_colors.append(None)
    
    else:
        raise ValueError(f"Invalid color_by value: {color_by}")

    # Define node radius in data coords (depends on node_size)
    radius = np.sqrt(node_size) / 200.0  # empirical scale factor

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis('off')

    # Draw edges
    for e in graph.es:
        src, tgt = e.source, e.target
        x0, y0 = layout[src]
        x1, y1 = layout[tgt]

        p0 = np.array((x0, y0))
        p1 = np.array((x1, y1))
        direction = p1 - p0
        length = np.linalg.norm(direction)
    
        # Normalize direction
        unit_dir = direction / length
    
        # Offset by node radius
        offset = radius * 1.2  # slightly more than radius to fully clear node
        new_start = p0 + unit_dir * offset
        new_end = p1 - unit_dir * offset

        arrow = FancyArrowPatch((x0, y0), (new_end), connectionstyle='arc3,rad=0.2',
                        arrowstyle='-|>', color='gray', lw=max(0.25, e['weight'] * 50), alpha=0.8,
                        mutation_scale=15)  # or try 25/30 if still small
        
        if highlighted_edges and (src, tgt) in highlighted_edges:
            arrow.set_color('limegreen')
            arrow.set_alpha(1)

        ax.add_patch(arrow)

    # Draw nodes
    for i, (x, y) in enumerate(layout):
        if terminal_clusters and i in terminal_clusters:
            edgecolor="#0c16ea"
            lw=2
        else: 
            edgecolor='black'
            lw=0.5
        if i in pie_data:
            if color_by == 'mut_cond' and any('TIC' in val for val in annotations[color_by].values):
                palette = {k.replace('A1 ', 'TIC'): v for k, v in palette.items()}
            start = 0
            for label, frac in pie_data[i].items():
                wedge = Wedge((x, y), radius, start * 360, (start + frac) * 360,
                              facecolor=palette[label], edgecolor='black', linewidth=0.5,)
                ax.add_patch(wedge)
                start += frac
            if terminal_clusters and i in terminal_clusters:
                circle = plt.Circle((x, y), radius+0.005, color=edgecolor, fill=False, lw=2)
                ax.add_artist(circle)
            
        else:
            ax.scatter(x, y, s=node_size, color=vertex_colors[i], edgecolors=edgecolor, zorder=3, lw=lw)
        ax.text(x + radius * 1.8, y + radius * 1.8, str(i+1), fontsize=12, ha='center', va='center', zorder=4, color='black')

    if color_by == 'mut_cond':
        # Increase figure size to accommodate the legend
        fig.set_size_inches(12, 10)
        # Create a legend for the mut_cond colors
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[label], markersize=8, label=label)
               for label in palette.keys()]
        ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5), title='Mutation Condition')
    
    elif color_by == 'pseudotime':
        # Create a colorbar for the pseudotime
        sm = plt.cm.ScalarMappable(cmap=palette, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.03, pad=0.02)
        cbar.set_label('Pseudotime', rotation=90, labelpad=15)
        cbar.ax.tick_params(labelsize=10)

    # Fix axis limits so nothing gets cut off
    x_vals = [pos[0] for pos in layout]
    y_vals = [pos[1] for pos in layout]
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)
    x_pad = (x_max - x_min) * 0.1
    y_pad = (y_max - y_min) * 0.1
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_aspect("equal")

    if not title:
        plt.title(f"Graph Colored by {color_by}", fontsize=16)
    else:
        plt.title(title, fontsize=16)
    plt.tight_layout()
    
    return fig




