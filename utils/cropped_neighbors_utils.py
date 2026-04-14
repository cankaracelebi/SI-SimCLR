import pandas as pd
import random
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as tf
import os


def cropped_neighbors(df: pd.DataFrame, index: int) -> pd.DataFrame:
    """
    Returns dataframe with some information about the cropped neighbors
    of a sample specified by the index argument. 
    """
    if 'hash' in df.columns:
        df = df.rename(columns={'hash': 'path'})
    elif 'img_code' in df.columns:
        df = df.rename(columns={'img_code': 'path'})
    else:
        raise KeyError('The "hash" or "img_code" columns are missing from your dataset annotation file.')
    
    item = df.iloc[index]
    #item = df.iloc[index]
    fov = item['fov']
    experiment = item['experiment']
    plate = item['plate']
    #experiment = item['experiment']
    row = item['row']
    col = item['col']
    #crop = item['crop']
    neighbors = df[(df['fov'] == fov) & (df['experiment'] == experiment) & (df['plate'] == plate) & (df['row'] == row) & (df['col'] == col)]
    columns_to_keep = ['index', 'fov', 'crop', 'path']
    columns_to_drop = list(set(neighbors.columns) - set(columns_to_keep))
    neighbors = neighbors.drop(columns_to_drop, axis=1)
    neighbors = neighbors.drop(index, axis=0)
    return neighbors


def select_random_cropped_neighbor(df: pd.DataFrame, transform=None):
    # Returns a torch.Tensor of size [3, H, W]
    neighbors_indices = df.index.values
    selected_neighbor_index = random.choice(neighbors_indices)
    neighbor_path = df.at[selected_neighbor_index, 'path']
    neighbor_image = Image.open(neighbor_path)
    return neighbor_image, selected_neighbor_index


def select_specific_cropped_neighbor(df: pd.DataFrame, index, transform=None):
    # Returns a torch.Tensor of size [3, 540, 540]
    neighbor_path = df.at[index, 'path']
    neighbor_image = Image.open(neighbor_path)
    #tensor_conversion = tf.Compose([tf.ToTensor()]) #Use ToTensor so the range of the entries is the same as the 'main' image
    #neighbor_image = tensor_conversion(neighbor_image)
    return neighbor_image, index


def adjacent_cropped_neighbors(df: pd.DataFrame, index: int) -> pd.DataFrame:
    """
    Returns dataframe with some information about the cropped neighbors
    of a sample specified by the index argument. 
    """
    if 'hash' in df.columns:
        df = df.rename(columns={'hash': 'path'})
    elif 'img_code' in df.columns:
        df = df.rename(columns={'img_code': 'path'})
    else:
        raise KeyError('The "hash" or "img_code" columns are missing from your dataset annotation file.')
    
    if df.crop.max() < 5:
        return cropped_neighbors(df, index)
    else:
        adjacent_neighbors = {1:  [2, 5, 6],
                              2:  [1, 3, 5, 6, 7],
                              3:  [2, 4, 6, 7, 8],
                              4:  [3, 7, 8],
                              5:  [1, 2, 6, 9, 10],
                              6:  [1, 2, 3, 5, 7, 9, 10, 11],
                              7:  [2, 3, 4, 6, 8, 10, 11, 12],
                              8:  [3, 4, 7, 11, 12],
                              9:  [5, 6, 10, 13, 14],
                              10: [5, 6, 7, 9, 11, 13, 14, 15],
                              11: [6, 7, 8, 10, 12, 14, 15, 16],
                              12: [7, 8, 11, 15, 16],
                              13: [9, 10, 14],
                              14: [9, 10, 11, 13, 15],
                              15: [10, 11, 12, 14, 16],
                              16: [11, 12, 15]}
        item = df.iloc[index]
        #item = df.iloc[index]
        fov = item['fov']
        experiment = item['experiment']
        plate = item['plate']
        #experiment = item['experiment']
        row = item['row']
        col = item['col']
        #crop = item['crop']
        neighbors = df[(df['fov'] == fov) & (df['experiment'] == experiment) & (df['plate'] == plate) & (df['row'] == row) & (df['col'] == col)]
        columns_to_keep = ['index', 'fov', 'crop', 'path']
        columns_to_drop = list(set(neighbors.columns) - set(columns_to_keep))
        neighbors = neighbors.drop(columns_to_drop, axis=1)
        neighbors = neighbors.drop(index, axis=0)
        neighbors = neighbors[neighbors.crop.isin(adjacent_neighbors[item['crop']])]
        return neighbors
