"""
Author: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
Modifications by Elisa Messori: added __meanitem__ and __stditem__ 
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils.cropped_neighbors_utils import select_random_cropped_neighbor, select_specific_cropped_neighbor
from utils.utils import custom_cartesian_product
from utils.annotations_utils import add_mutation_column
import itertools
from PIL import Image

""" 
    AugmentedDataset
    Returns an image together with an augmentation.
"""
class AugmentedDataset(Dataset):
    def __init__(self, dataset):
        super(AugmentedDataset, self).__init__()
        transform = dataset.transform
        dataset.transform = None
        self.dataset = dataset
        
        if isinstance(transform, dict):
            self.image_transform = transform['standard']
            self.augmentation_transform = transform['augment']

        else:
            self.image_transform = transform
            self.augmentation_transform = transform

    def __len__(self):
        return len(self.dataset)
        

    def __getitem__(self, index):
        sample = self.dataset.__getitem__(index)
        image = sample['image']
        
        sample['image'] = self.image_transform(image)
        sample['image_augmented'] = self.augmentation_transform(image)

        return sample


    def __meanitem__(self, index):
        mean = []
        sample = self.dataset.__getitem__(index)
        totensor = transforms.Compose([
        transforms.ToTensor()
        ])
        image = totensor(sample['image'])
        r = (image.permute(1,2,0))[:,:,0]
        g = (image.permute(1,2,0))[:,:,1]
        b = (image.permute(1,2,0))[:,:,2]
        r_mean = r.mean()
        mean.append(float(r_mean))
        g_mean = g.mean()
        mean.append(float(g_mean))
        b_mean = b.mean()
        mean.append(float(b_mean))

        return mean


    def __stditem__(self, index):
        std = []
        sample = self.dataset.__getitem__(index)
        totensor = transforms.Compose([
        transforms.ToTensor()
        ])
        image = totensor(sample['image'])
        r = (image.permute(1,2,0))[:,:,0]
        g = (image.permute(1,2,0))[:,:,1]
        b = (image.permute(1,2,0))[:,:,2]
        r_std = r.std()
        std.append(float(r_std))
        g_std = g.std()
        std.append(float(g_std))
        b_std = b.std()
        std.append(float(b_std))

        return std


"""
    CroppedNeighborDataset
    To be used during training of the cropped neighbor variant
    Defined by modifying the AugmentedDataset class above
    By Elisa Messori
"""

class CroppedNeighborDataset(Dataset):
    def __init__(self, dataset):
        super(CroppedNeighborDataset, self).__init__()
        self.transform = dataset.transform
        dataset.transform = None
        self.dataset = dataset
        self.dataframe = dataset.dataframe
        self.paired_samples = []
        self.repeated_pairs = 0

        if 'img_code' in self.dataframe.columns:
            self.dataframe = self.dataframe.rename(columns={'img_code': 'path'})
        elif 'hash' in self.dataframe.columns:
            self.dataframe = self.dataframe.rename(columns={'hash': 'path'})
        
        # BE CAREFUL WITH THE TRANSFORMS
        if isinstance(self.transform, dict):
            self.image_transform = self.transform['standard']
            self.augmentation_transform = self.transform['augment']
        else:
            self.image_transform = self.transform
            self.augmentation_transform = self.transform

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, index):
        attempt = 0
        sample = self.dataset.__getitem__(index)
        neighbors_indices = list(sample['cropped_neighbors'].index)
        paired_samples = self.paired_samples
        need_constraint = False
        neighbors_indices.append(index)

        all_possible_pairs_ = itertools.permutations(neighbors_indices, 2)
        all_possible_pairs = [pair for pair in all_possible_pairs_]
        pairs_to_check = [pair for pair in all_possible_pairs if pair in paired_samples]
        for tmp_index in neighbors_indices[:-1]:
            occ = 0
            for pair in pairs_to_check:
                if tmp_index == pair[1]:
                    occ += 1
                    #pass
            if occ>1:
                need_constraint = True
                index_to_remove = tmp_index
                neighbors_indices.remove(index_to_remove)
        neighbors_indices.remove(index)
        possible_pairs = custom_cartesian_product([index], neighbors_indices)
        allowed_pairs = [ x for x in possible_pairs if (x not in self.paired_samples) and ((x[1], x[0]) not in self.paired_samples)]
        image = sample['image']
        sample['image'] = self.image_transform(image)
        neighbors_df = sample['cropped_neighbors']
        if len(pairs_to_check)==1:
            neighbor_specific_index = pairs_to_check[0][0]
            ### CHECK IF THAT PAIR IS ALREADY PRESENT
            if not((neighbor_specific_index, index) in self.paired_samples):
                selected_neighbor = select_specific_cropped_neighbor(neighbors_df, neighbor_specific_index)
                self.paired_samples.append((index, selected_neighbor[1]))
                sample['image_augmented'] = self.augmentation_transform(selected_neighbor[0])
                return sample
        duplicate = True
        if len(neighbors_indices)==1:
            selected_neighbor = select_random_cropped_neighbor(neighbors_df)
            sample['image_augmented'] = self.augmentation_transform(selected_neighbor[0])
            self.paired_samples.append((index, selected_neighbor[1]))
            self.repeated_pairs += 1
            return sample

        while duplicate and (len(allowed_pairs)>0):
            attempt += 1
            selected_neighbor = select_random_cropped_neighbor(neighbors_df)
            if ((index, selected_neighbor[1]) in allowed_pairs):
                self.paired_samples.append((index, selected_neighbor[1]))
                duplicate = False
        if (len(allowed_pairs)==0) and duplicate:
            pairs_with_index = [pair for pair in self.paired_samples if index in pair]
            print(f'Pairs that contain the "faulty" index: {pairs_with_index}')
            raise RuntimeError(f'For image {index}, there are no available pairs.')
        #
        # I used the same key name as in the augmentation pairing strategy s.t. I don't have to change names in the simcrl_train function.
        sample['image_augmented'] = self.augmentation_transform(selected_neighbor[0])

        return sample
    
class MutCondDataset(CroppedNeighborDataset):
        def __init__(self, dataset):
            super(MutCondDataset, self).__init__(dataset)
            self.transform = dataset.transform
            dataset.transform = None
            self.dataset = dataset
            self.dataframe = dataset.dataframe
            self.dataframe = self.dataframe[~self.dataframe['cell_line'].str.contains('SOD1')]
            self.dataframe = self.dataframe.set_index(np.arange(self.dataframe.shape[0]))
            self.paired_samples = []
            self.repeated_pairs = 0

            if 'img_code' in self.dataframe.columns:
                self.dataframe = self.dataframe.rename(columns={'img_code': 'path'})
            elif 'hash' in self.dataframe.columns:
                self.dataframe = self.dataframe.rename(columns={'hash': 'path'})

            if isinstance(self.transform, dict):
                self.image_transform = self.transform['standard']
                self.augmentation_transform = self.transform['augment']
            else:
                self.image_transform = self.transform
                self.augmentation_transform = self.transform

            self.dataframe['exp_plate'] = self.dataframe['experiment'] + '_' + self.dataframe['plate']

        def __getitem__(self, index):
            sample = self.dataset.__getitem__(index)
            mut = self.dataframe.loc[index]['mutation']
            cond = self.dataframe.loc[index]['condition']
            #exp_plate = self.dataframe.loc[index]['exp_plate']
            experiment = self.dataframe.loc[index]['experiment']

            temp_df = self.dataframe.copy()
            temp_df = temp_df[(temp_df['mutation'] == mut) & (temp_df['condition'] == cond)]
            temp_df = temp_df[temp_df['experiment'] != experiment]
            mutcond_indices = temp_df.index.to_list()

            if not mutcond_indices:
                print(f'No other images with the same mutation and condition as image {index}.')
                print(f'Mutation: {mut}, condition: {cond}, batch (day): {experiment}')
                raise RuntimeError(f'No other images with the same mutation and condition as image {index}.')

            # Let's try to avoid having the same sample too many times in the positive pairs
            filtered_indices = []
            for i in mutcond_indices:
                count = 0
                for p in self.paired_samples:
                    if i in p:
                        count += 1
                if count <= 10:
                    filtered_indices.append(i)

            mutcond_indices = filtered_indices

            if not mutcond_indices:
                print(f'No valid images with the same mutation and condition as image {index} after filtering.')
                raise RuntimeError(f'No valid images with the same mutation and condition as image {index} after filtering.')

            # Select a random index
            positive_index = np.random.choice(mutcond_indices, 1)[0]
            # Check if (positive_index, index) is already in paired_samples
            while (positive_index, index) in self.paired_samples:
                positive_index = np.random.choice(mutcond_indices, 1)[0]

            positive_path = self.dataframe.at[index, 'path']
            positive_image = Image.open(positive_path)
            image = sample['image']
            sample['image'] = self.image_transform(image)
            sample['image_augmented'] = self.augmentation_transform(positive_image)
            self.paired_samples.append((index, positive_index))
            return sample
    

class AdjacentNeighborDataset(CroppedNeighborDataset):

    def __getitem__(self, index):
        attempt = 0
        sample = self.dataset.__getitem__(index)
        neighbors_indices = list(sample['adjacent_neighbors'].index)
        meta = sample['meta']
        crop = meta['crop']
        paired_samples = self.paired_samples
        need_constraint = False
        neighbors_indices.append(index)

        all_possible_pairs_ = itertools.permutations(neighbors_indices, 2)
        all_possible_pairs = [pair for pair in all_possible_pairs_]
        pairs_to_check = [pair for pair in all_possible_pairs if pair in paired_samples]
        for tmp_index in neighbors_indices[:-1]:
            occ = 0
            for pair in pairs_to_check:
                if tmp_index == pair[1]:
                    occ += 1
                    #pass
            if occ>2:
                need_constraint = True
                index_to_remove = tmp_index
                neighbors_indices.remove(index_to_remove)
        neighbors_indices.remove(index)
        
        neighbors_neighbors = dict(zip(neighbors_indices, [0 for n in neighbors_indices]))
        for neighbor_index in neighbors_indices:
            neighbor_sample = self.dataset.__getitem__(neighbor_index)
            neighbor_neighbors = list(neighbor_sample['adjacent_neighbors'].index)
            #neighbor_neighbors.remove(neighbor_index)
            neighbors_neighbors[neighbor_index] = len(neighbor_neighbors)
        
        count_above_threshold = sum(1 for value in neighbors_neighbors.values() if value >1)
        if len(neighbors_indices)>1 and count_above_threshold>0:
            neighbors_indices = [index for index, count in neighbors_neighbors.items() if count >1]

        possible_pairs = custom_cartesian_product([index], neighbors_indices)
        allowed_pairs = [ x for x in possible_pairs if (x not in self.paired_samples) and ((x[1], x[0]) not in self.paired_samples)]
        image = sample['image']
        sample['image'] = self.image_transform(image)
        neighbors_df = self.dataframe.loc[neighbors_indices]
        if len(pairs_to_check)==1:
            neighbor_specific_index = pairs_to_check[0][0]
            ### CHECK IF THAT PAIR IS ALREADY PRESENT
            if not((neighbor_specific_index, index) in self.paired_samples):
                selected_neighbor = select_specific_cropped_neighbor(neighbors_df, neighbor_specific_index)
                self.paired_samples.append((index, selected_neighbor[1]))
                sample['image_augmented'] = self.augmentation_transform(selected_neighbor[0])
                return sample
        duplicate = True
        if len(neighbors_indices)==1:
            selected_neighbor = select_random_cropped_neighbor(neighbors_df)
            sample['image_augmented'] = self.augmentation_transform(selected_neighbor[0])
            if (selected_neighbor[1], index) in self.paired_samples:
                self.repeated_pairs += 1
                if self.repeated_pairs % 10 == 0:
                    print('We currently have {} repeated pairs.'.format(self.repeated_pairs))
            self.paired_samples.append((index, selected_neighbor[1]))
            return sample

        while duplicate and (len(allowed_pairs)>0):
            attempt += 1
            selected_neighbor = select_random_cropped_neighbor(neighbors_df)
            if ((index, selected_neighbor[1]) in allowed_pairs):
                self.paired_samples.append((index, selected_neighbor[1]))
                duplicate = False
        if (len(allowed_pairs)==0) and duplicate:
            pairs_with_index = [pair for pair in self.paired_samples if index in pair]
            #print(f'!!! Detected crop for which there are no new original pairs: {index} !!!')
            #print(f'Pairs that contain the "faulty" index: {pairs_with_index}')
            #print(f'Adjacent neighbors for image {index} (crop {crop}): {neighbors_indices}')
            # Select a random neighbor among neighbors_indices
            selected_neighbor = select_random_cropped_neighbor(neighbors_df)
            self.paired_samples.append((index, selected_neighbor[1]))
            print(f'Randomly selected adjacent neighbor: {selected_neighbor[1]}')
            self.repeated_pairs += 1
            if self.repeated_pairs % 10 == 0:
                print('We currently have {} repeated pairs.'.format(self.repeated_pairs))
            
            #raise RuntimeError(f'For image {index} (crop {crop}), there are no available pairs.')
        #
        # I used the same key name as in the augmentation pairing strategy s.t. I don't have to change names in the simcrl_train function.
        sample['image_augmented'] = self.augmentation_transform(selected_neighbor[0])

        return sample


""" 
    NeighborsDataset (for SCAN clustering)
    Returns an image with one of its neighbors.
"""
class NeighborsDataset(Dataset):
    def __init__(self, dataset, indices, num_neighbors=None):
        super(NeighborsDataset, self).__init__()
        transform = dataset.transform
        if isinstance(transform, dict):
            self.anchor_transform = transform['standard']
            self.neighbor_transform = transform['augment']
        else:
            self.anchor_transform = transform
            self.neighbor_transform = transform
       
        dataset.transform = None
        self.dataset = dataset
        self.indices = indices # Nearest neighbor indices (np.array  [len(dataset) x k])
        if num_neighbors is not None:
            self.indices = self.indices[:, :num_neighbors+1]
        assert(self.indices.shape[0] == len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = {}
        anchor = self.dataset.__getitem__(index)
        
        neighbor_index = np.random.choice(self.indices[index], 1)[0]
        neighbor = self.dataset.__getitem__(neighbor_index)

        anchor['image'] = self.anchor_transform(anchor['image'])
        neighbor['image'] = self.neighbor_transform(neighbor['image'])

        output['anchor'] = anchor['image']
        output['neighbor'] = neighbor['image'] 
        output['possible_neighbors'] = torch.from_numpy(self.indices[index])
        output['target'] = anchor['target']
        
        return output
