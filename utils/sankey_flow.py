from collections import defaultdict
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import os
import plotly.graph_objects as go
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np



def to_rgba_color(val, opacity=0.5):
        try:
            rgba = mcolors.to_rgba(val, alpha=opacity)
            r, g, b, a = [int(255*x) if i < 3 else x for i, x in enumerate(rgba)]
            return f'rgba({r},{g},{b},{a:.2f})'
        except Exception:
            if isinstance(val, str) and val.startswith('#') and len(val) == 7:
                rgba = mcolors.to_rgba(val, alpha=opacity)
                r, g, b, a = [int(255*x) if i < 3 else x for i, x in enumerate(rgba)]
                return f'rgba({r},{g},{b},{a:.2f})'
            return 'rgba(128,128,128,0.5)'  # fallback gray
        


def plot_sankey_flow(transition_data, cluster_palette: dict,
                     size_threshold=0.0, edge_threshold=0.05,
                     title=None, opacity=0.5,
                     source_to_display=None, target_to_display=None,
                     saving_dir = None,
                     save_name=None, color_edge_according_to='source'):


    transition_matrix = transition_data['transition_matrix']
    source_classes = list(transition_data['classes_1'])
    target_classes = list(transition_data['classes_2'])
    annotations = transition_data['annotations_df']
    category_column = transition_data['category_column']
    celltype_column = transition_data['celltype_column']
    source = transition_data['source']
    target = transition_data['target']

    # Normalize transition matrix
    #transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0)

    source_ann = annotations[annotations[category_column] == source]
    target_ann = annotations[annotations[category_column] == target]

    total_source_cells = len(source_ann)
    total_target_cells = len(target_ann)
    source_sizes = {
        cluster: len(source_ann[source_ann[celltype_column] == cluster]) / total_source_cells
        for cluster in source_ann[celltype_column].unique()
    }
    target_sizes = {
        cluster: len(target_ann[target_ann[celltype_column] == cluster]) / total_target_cells
        for cluster in target_ann[celltype_column].unique()
    }

    # Filter by size threshold
    source_classes = [c for c in source_classes if source_sizes.get(c, 0) >= size_threshold]
    target_classes = [c for c in target_classes if target_sizes.get(c, 0) >= size_threshold]
    transition_matrix = transition_matrix.loc[source_classes, target_classes]

    # Filter by source_to_display and target_to_display if provided
    if source_to_display is not None:
        if source_to_display not in source_classes:
            raise ValueError(f"Source class '{source_to_display}' not found among source clusters.")
        source_classes = [source_to_display]
        transition_matrix = transition_matrix.loc[source_classes]
    if target_to_display is not None:
        if target_to_display not in target_classes:
            raise ValueError(f"Target class '{target_to_display}' not found among target clusters.")
        target_classes = [target_to_display]
        transition_matrix = transition_matrix[target_classes]

    # Node labels: only real nodes (no invisible nodes)
    node_labels = [str(c) for c in source_classes] + ["" for _ in target_classes]
    node_colors = [to_rgba_color(cluster_palette.get(c, "gray"), 1.0) for c in source_classes] + [to_rgba_color(cluster_palette.get(c, "gray"), 1.0) for c in target_classes]

    # Sankey links:
    sources = []
    targets = []
    values = []
    link_colors = []
    present_source_indices = set()
    present_target_indices = set()
    for i, src in enumerate(source_classes):
        for j, tgt in enumerate(target_classes):
            val = transition_matrix.loc[src, tgt]
            if val > edge_threshold:
                sources.append(i)
                targets.append(len(source_classes) + j)
                values.append(val)
                present_source_indices.add(i)
                present_target_indices.add(j)
                if color_edge_according_to == 'target':
                    color_val = cluster_palette.get(tgt, "gray")
                elif color_edge_according_to == 'source':
                    color_val = cluster_palette.get(src, "gray")
                link_colors.append(to_rgba_color(color_val, opacity))

    # Build Sankey figure (no extra nodes)
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=[""] * len(node_labels),  # hide built-in labels
            color=node_colors,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
        )
    )])

    fig.update_layout(title_text=title or "Sankey Flow Graph", font_size=12, width=720, height=1080)

    n_source = len(source_classes)
    n_target = len(target_classes)

    # Approximate node positions manually:
    y_sources = np.linspace(0.9, 0.1, n_source) if n_source > 1 else [0.5]
    y_targets = np.linspace(0.9, 0.1, n_target) if n_target > 1 else [0.5]

    annotations = []
    # Left-side labels (sources, in node order)
    #for i, y in enumerate(y_sources):
    #    if i in present_source_indices:
    #        src = source_classes[i]
    #        annotations.append(dict(
    #            x=-0.05,  # left edge
    #            y=y,
    #            xref="paper",
    #            yref="paper",
    #            text=str(src),
    #            showarrow=False,
    #            font=dict(size=14, color="black"),
    #            xanchor="right",
    #            yanchor="middle",
    #        ))

    # Right-side labels (targets, in node order)
    #for j, y in enumerate(y_targets):
    #    if j in present_target_indices:
    #        tgt = target_classes[j]
    #        annotations.append(dict(
    #            x=1.05,  # right edge
    #            y=y,
    #            xref="paper",
    #            yref="paper",
    #            text=str(tgt),
    #            showarrow=False,
    #            font=dict(size=14, color="black"),
    #            xanchor="left",
    #            yanchor="middle",
    #        ))

    fig.update_layout(annotations=annotations)
    #if save_name == None:
    #    fig.write_image(os.path.join(saving_dir, f"sankey_flow_{source}_to_{target}.svg"))
    if (save_name != None) and (saving_dir != None):
        fig.write_image(os.path.join(saving_dir, save_name))
    fig.show()