import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_transition_matrix(transition_matrix_computation_output, cluster_palette, title=None, figsize=(12, 8)):
    """
    Plots a state transition matrix with colored axis labels.

    Parameters:
    - results: dict, output from compute_state_transition_matrix
    - cluster_palette: dict, mapping cluster labels to colors
    - title: str, plot title
    - figsize: tuple, figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(transition_matrix_computation_output['transition_matrix'].values, cmap='viridis')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label('Transition Probability', labelpad=15)

    # Set axis labels with category names
    ax.set_xticks(np.arange(len(transition_matrix_computation_output['transition_matrix'].columns)))
    ax.set_yticks(np.arange(len(transition_matrix_computation_output['transition_matrix'].index)))
    ax.set_xticklabels(
        transition_matrix_computation_output['transition_matrix'].columns,
        ha='center', 
        fontweight='bold', fontsize=14
    )
    ax.set_yticklabels(
        transition_matrix_computation_output['transition_matrix'].index, fontweight='bold', fontsize=14
    )

    for tick in ax.get_xticklabels():
        try:
            tick.set_color(cluster_palette.get(int(tick.get_text()), 'black'))
        except ValueError:
            tick.set_color('black')
    for tick in ax.get_yticklabels():
        try:
            tick.set_color(cluster_palette.get(int(tick.get_text()), 'black'))
        except ValueError:
            tick.set_color('black')

    # Add axis labels
    ax.set_xlabel(transition_matrix_computation_output['target'], fontweight='bold', fontsize=14)
    ax.set_ylabel(transition_matrix_computation_output['source'], fontweight='bold', fontsize=14)

    # Add title
    if not (title is None):
        ax.set_title(title)

    plt.tight_layout()
    return fig, ax