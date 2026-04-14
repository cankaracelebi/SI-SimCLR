"""
Loading of astrocytes dataset
Author: Elisa Messori
"""

import os
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from utils.mypath import MyPath
from torchvision import transforms as tf
import torchvision.datasets as datasets
import pandas as pd
from glob import glob
import numpy as np
from utils.cropped_neighbors_utils import cropped_neighbors, adjacent_cropped_neighbors


class EF_Crop16_AC_2405(Dataset):
    def __init__(self, root=MyPath.db_root_dir('crop16_EF_2405'), transform=None, image_root=None):
        self.transform = transform
        self.image_root = image_root
        self.dataframe = pd.read_csv(root)
        if 'img_code' in self.dataframe.columns and self.image_root is not None:
            self.dataframe['img_code'] = self.dataframe['img_code'].apply(self._resolve_image_path)

    def _resolve_image_path(self, img_code):
        if not isinstance(img_code, str):
            return img_code
        if os.path.isabs(img_code):
            return img_code
        return os.path.normpath(os.path.join(self.image_root, img_code))

    def __getitem__(self, index):
        # Returns dataset item, i.e. transformed image with neighbors etc
        item = self.dataframe.iloc[index]
        img_path = item['img_code']
        crop = item['crop']
        target = 0 #MODIFY THIS IF YOU WANT TO PERFORM VALIDATION WITH LABELLED DATA
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        im_size = img.size

        als_label = item['als_label']
        treatment = item['condition']
        cell_line = item['cell_line']

        if self.transform is not None:
            img = self.transform(img)
        out = {'image': img,
               'target': target,
               'meta': {'im_size': im_size, 'index': index, 'crop': crop},
               'labels': {'als_label': als_label, 'treatment': treatment, 'cell_line': cell_line},
               'cropped_neighbors': cropped_neighbors(self.dataframe, index),
               'adjacent_neighbors': adjacent_cropped_neighbors(self.dataframe, index)
               }
        return out

    def __len__(self):
        return len(self.dataframe)

    def get_image(self, index):
        # Returns original image, without transformations
        item = self.dataframe.iloc[index]
        img_path = item['img_code']
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        return img

    def als_labels(self):
        als_labels_ = []
        for index, row in self.dataframe.iterrows():
            als_labels_.append(row['als_label'])
        return als_labels_

    def treatment_labels(self):
        treatment_labels_ = []
        for index, row in self.dataframe.iterrows():
            treatment_labels_.append(row['condition'])
        return treatment_labels_

