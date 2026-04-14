import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import seaborn as sns

# Define 4 equidistant colors from matplotlib palette 'summer']
SOD1_palette = cm.get_cmap('Purples', 5)
SOD1_palette = [matplotlib.colors.to_hex(SOD1_palette(i)) for i in range(5)]
VCP_palette = cm.get_cmap('YlOrBr', 4)
VCP_palette = [matplotlib.colors.to_hex(VCP_palette(i)) for i in range(5)]
BATCH_palette = sns.color_palette('Dark2', 8)
BATCH_palette = [matplotlib.colors.to_hex(BATCH_palette[i]) for i in range(8)]

magenta = '#D60078'

pairing_strategy_palette = {
    'Hand-crafted': '#c0c0c0',                    
    'Pretrained ResNet18': '#d781af',            
    'Standard SimCLR': '#829fd0',                        
    'Augmented SimCLR': '#BCA337',              
    'uniDINO': "#37A85D",
    'CellPaint SSL': "#025E21",
    'hand-crafted': '#c0c0c0',
    'Pretrained\nResNet18': '#d781af',
    'SI SimCLR': '#BCA337'
}

# Define new cell_line_palette with CTRL from Greens and VCP from RdPu
greens_cmap = cm.get_cmap('Greens', 6)
rdpu_cmap = cm.get_cmap('RdPu', 7)
cell_line_palette = {
    'CTRL1': matplotlib.colors.to_hex(greens_cmap(2)),
    'CTRL2': matplotlib.colors.to_hex(greens_cmap(3)),
    'CTRL4': matplotlib.colors.to_hex(greens_cmap(4)),
    'CTRL5': matplotlib.colors.to_hex(greens_cmap(5)),
    'GLIA': matplotlib.colors.to_hex(rdpu_cmap(2)),
    'GLIB': matplotlib.colors.to_hex(rdpu_cmap(3)),
    'CBI': matplotlib.colors.to_hex(rdpu_cmap(4)),
    'CBID': matplotlib.colors.to_hex(rdpu_cmap(5)),
    'CBIE': matplotlib.colors.to_hex(rdpu_cmap(6)),
}

mutation_palette = {'CTRL': '#1b5e20', 'VCP': magenta, 'SOD1': SOD1_palette[3], 'ALS': magenta}

als_palette = {'control': mutation_palette['CTRL'], 'als': mutation_palette['VCP'],
               'CTRL': mutation_palette['CTRL'], 'ALS': mutation_palette['VCP']}

exp_plate_palette = {'E1_P1': '#021aca', 'E2_P1': BATCH_palette[1], 'E2_P2': BATCH_palette[2],
                     'E3_P1': BATCH_palette[3], 'E4_P1': BATCH_palette[4], 'E4_P2': BATCH_palette[5]}

condition_palette = {'untreated': '#424949', 'A1 low': '#00b0ff', 'A1 high': '#ff8b60',
                     'TIClow': '#00b0ff', 'TIChigh': '#ff8b60',}

day_palette = {'E1': '#021aca', 'E2': BATCH_palette[1], 'E3': BATCH_palette[3], 'E4': BATCH_palette[4]}

factors_palette = {'CTRL_untreated': condition_palette['untreated'],
                   'CTRL_A1 low': condition_palette['A1 low'],
                   'CTRL_A1 high': condition_palette['A1 high'],
                   'VCP_untreated': mutation_palette['VCP'],
                   'VCP_A1 low': condition_palette['A1 low'],
                   'VCP_A1 high': condition_palette['A1 high'],}

cmap_mutation_colors = [(0.106, 0.369, 0.125),
    (0.278, 0.573, 0.298),
    (0.553, 0.749, 0.569),
    (0.937, 0.624, 0.800),
    (1.000, 0.376, 0.725),
    (0.839, 0.000, 0.471)]

cmap_mutation = LinearSegmentedColormap.from_list('Random gradient 9670', (
    # Edit this gradient at https://eltos.github.io/gradient/#Random%20gradient%209670=1B5E20-47924C-8DBF91-EF9FCC-FF60B9-D60078
    (0.000, (0.106, 0.369, 0.125)),
    (0.200, (0.278, 0.573, 0.298)),
    (0.400, (0.553, 0.749, 0.569)),
    (0.600, (0.937, 0.624, 0.800)),
    (0.800, (1.000, 0.376, 0.725)),
    (1.000, (0.839, 0.000, 0.471))))

label_order = ['CTRL_A1 high', 'CTRL_A1 low', 'CTRL_untreated', 'VCP_untreated', 'VCP_A1 low', 'VCP_A1 high']
palette_mutcond = dict(zip(label_order, cmap_mutation_colors))
palette_mutcond['CTRL_TIClow'] = palette_mutcond['CTRL_A1 low']
palette_mutcond['CTRL_TIChigh'] = palette_mutcond['CTRL_A1 high']

def cluster_palette(n_clusters):
    if n_clusters <= 20:
        cl_palette = sns.color_palette('tab20', n_clusters)
    else:
        cl_palette = sns.color_palette('tab20', 20) + sns.color_palette('husl', n_clusters - 20)
        
    return cl_palette


def cluster_palette_dict(labels):
    n_clusters = len(set(labels))
    if n_clusters <= 20:
        cl_palette = sns.color_palette('tab20', n_clusters)
    else:
        cl_palette = sns.color_palette('tab20', 20) + sns.color_palette('husl', n_clusters - 20)

    cl_palette_dict = {l: cl_palette[i] for i, l in enumerate(set(labels))}
    
    return cl_palette_dict