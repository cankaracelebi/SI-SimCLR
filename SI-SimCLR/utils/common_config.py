"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
Modifications by Elisa Messori: added everything related to the astrocytes dataset (datasets loading, trasnformations 
containing the term 'acs')
"""
import os
import math
import numpy as np
import torch
#import albumentations as A
#from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms
from data.augment import Augment, Cutout
from utils.collate import collate_custom
import torchvision.transforms.functional as TF
from PIL import Image
import random
from typing import Sequence


class Rotate90deg:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


class Permute_Channels:
    def __init__(self, channels: Sequence[str]):
        self.channels = channels

    def __call__(self, x):
        R, G, B = x.split()
        if (self.channels == ['B', 'G']) or (self.channels == ['G', 'B']):
            return Image.merge("RGB", [R, B, G])
        elif (self.channels == ['R', 'G']) or (self.channels == ['G', 'R']):
            return Image.merge("RGB", [G, R, B])
        elif (self.channel == ['R', 'B']) or (self.channels == ['B', 'R']):
            return Image.merge("RGB", [B, G, R])
        else:
            raise ValueError('Two channels must be provided as argument (i.e. [R, G] etc)')
        
 
def get_criterion(p):
    
    if p['criterion'] == 'simclr':
        from losses.losses import SimCLRLoss
        criterion = SimCLRLoss(**p['criterion_kwargs'])
    else:
        raise ValueError('Invalid criterion {}'.format(p['criterion']))

    return criterion


def get_feature_dimensions_backbone(p):
    if p['backbone'] == 'resnet18':
        return 512

    elif p['backbone'] == 'resnet50':
        return 2048

    else:
        raise NotImplementedError


def get_model(p, pretrain_path=None):
    # Get backbone
    if p['backbone'] == 'resnet18':
        if p['train_db_name'] in ['cifar-10', 'cifar-20']:
            from models.resnet_cifar import resnet18
            backbone = resnet18()

        elif p['train_db_name'] == 'stl-10':
            from models.resnet_stl import resnet18
            backbone = resnet18()

        elif p['train_db_name'] in ['crop16_EF_2405', 'crop16_EF_2405_adj_single']:
            from models.resnet_cifar import resnet18
            backbone = resnet18(p['backbone_dim'])
        
        else:
            raise NotImplementedError

    elif p['backbone'] == 'resnet50':
        if 'imagenet' in p['train_db_name']:
            from models.resnet import resnet50
        elif 'custom' in p['train_db_name']:
            from models.resnet import resnet50
            backbone = resnet50()

        else:
            raise NotImplementedError 

    else:
        raise ValueError('Invalid backbone {}'.format(p['backbone']))

    # Setup
    if p['setup'] in ['simclr', 'moco']:
        from models.models import ContrastiveModel
        model = ContrastiveModel(backbone, **p['model_kwargs'])

    else:
        raise ValueError('Invalid setup {}'.format(p['setup']))

    # Load pretrained weights
    if pretrain_path is not None and os.path.exists(pretrain_path):
        state = torch.load(pretrain_path, map_location='cpu')
        
        if p['setup'] == 'scan': # Weights are supposed to be transfered from contrastive training
            missing = model.load_state_dict(state, strict=False)
            assert(set(missing[1]) == {
                'contrastive_head.0.weight', 'contrastive_head.0.bias', 
                'contrastive_head.2.weight', 'contrastive_head.2.bias'}
                or set(missing[1]) == {
                'contrastive_head.weight', 'contrastive_head.bias'})

        else:
            raise NotImplementedError

    elif pretrain_path is not None and not os.path.exists(pretrain_path):
        raise ValueError('Path with pre-trained weights does not exist {}'.format(pretrain_path))

    else:
        pass

    return model


def get_pairing_strategy(p):
    """
    Returns a binary vector whose entries will be used as bool variable in get_train_dataset
    The entries refer to:
    1. Augmentation strategy (i.e. applying different transformations to the same image)
    2. Cropped Neighbor strategy (i.e. pairing an image and its cropped neighbor
       that before protocolacs come from the same 1080x1080 image)
    """
    pairing_instructions = {'augmentations': 0,
                            'adjacent_neighbors': 0,
                            }
    strategy = p['pairing_strategy']

    if strategy not in pairing_instructions.keys():
        raise ValueError('Invalid pairing strategy {}'.format(strategy))
    
    pairing_instructions[strategy] = 1
    
    return pairing_instructions


def get_train_dataset(p, transform,
                      to_augmented_dataset=False,
                      to_adjacent_neighbor_dataset=False,
                      split=None):

    if p['train_db_name'] in ['crop16_EF_2405', 'crop16_EF_2405_adj_single']:
        from data.acs_dataset import EF_Crop16_AC_2405
        dataset = EF_Crop16_AC_2405(transform=transform, image_root=p.get('image_root_dir', None))

    else:
        raise ValueError('Invalid train dataset {}'.format(p['train_db_name']))
    
    # Wrap into other dataset (__getitem__ changes)
    if to_augmented_dataset: # Dataset returns an image and an augmentation of that image.
        from data.custom_dataset import AugmentedDataset
        dataset = AugmentedDataset(dataset)

    if to_adjacent_neighbor_dataset:
        from data.custom_dataset import AdjacentNeighborDataset
        dataset = AdjacentNeighborDataset(dataset)
    
    #if to_neighbors_dataset: # Dataset returns an image and one of its nearest neighbors.
    #    from data.custom_dataset import NeighborsDataset
    #    indices = np.load(p['topk_neighbors_train_path'])
    #    dataset = NeighborsDataset(dataset, indices, p['num_neighbors'])
    
    return dataset


#def get_val_dataset(p, transform=None, to_cropped_neighbor_dataset=True, to_neighbors_dataset=False):
#    # Base dataset
#    
#    if p['val_db_name'] == 'crop16_ac_complete':
#        from data.acs_dataset import Crop16_AC_Complete
#        dataset = Crop16_AC_Complete(transform=transform)
#
#    elif p['train_db_name'] == 'crop4_EF_2405':
#        from data.acs_dataset import EF_Crop4_AC_2405
#        dataset = EF_Crop4_AC_2405(transform=transform)
#
#    elif p['train_db_name'] == 'crop16_EF_2405':
#        from data.acs_dataset import EF_Crop16_AC_2405
#        dataset = EF_Crop16_AC_2405(transform=transform)
#    
#    elif p['train_db_name'] == 'HDBS_EF_crop4_ac_nosingle':
#        from data.acs_dataset import HDBS_EF_Crop4_AC_NoSingle
#        dataset = HDBS_EF_Crop4_AC_NoSingle(transform=transform)
#
#    elif p['train_db_name'] == 'HDBS_EF_crop16_ac_nosingle':
#        from data.acs_dataset import HDBS_EF_Crop16_AC_NoSingle
#        dataset = HDBS_EF_Crop16_AC_NoSingle(transform=transform)
#
#    elif p['val_db_name'] == 'completeacs_unsplitted':
#        from data.acs_dataset import CompleteACS_unsplitted
#        dataset = CompleteACS_unsplitted(transform=transform)
#
#    elif p['val_db_name'] in ['imagenet_50', 'imagenet_100', 'imagenet_200']:
#        from data.imagenet import ImageNetSubset
#        subset_file = './data/imagenet_subsets/%s.txt' %(p['val_db_name'])
#        dataset = ImageNetSubset(subset_file=subset_file, split='val', transform=transform)
#    
#    else:
#        raise ValueError('Invalid validation dataset {}'.format(p['val_db_name']))
#    
#    if to_cropped_neighbor_dataset:
#        from data.custom_dataset import CroppedNeighborDataset
#        dataset = CroppedNeighborDataset(dataset)
#
#    # Wrap into other dataset (__getitem__ changes) 
#    if to_neighbors_dataset: # Dataset returns an image and one of its nearest neighbors.
#        from data.custom_dataset import NeighborsDataset
#        indices = np.load(p['topk_neighbors_val_path'])
#        dataset = NeighborsDataset(dataset, indices, 5) # Only use 5
#
#    return dataset


def get_train_dataloader(p, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'], 
            batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
            drop_last=True, shuffle=True)


def get_val_dataloader(p, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'],
            batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
            drop_last=False, shuffle=False)


def get_train_transformations(p):
    if p['augmentation_strategy'] == 'standard':
        # Standard augmentation strategy
        return transforms.Compose([
            transforms.RandomResizedCrop(**p['augmentation_kwargs']['random_resized_crop']),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize(**p['augmentation_kwargs']['normalize'])
        ])

    elif p['augmentation_strategy'] == 'simclr_acs':
        return transforms.Compose([
            transforms.RandomApply([
            transforms.Resize(p['augmentation_kwargs']['resize']['size'])],
            p=p['augmentation_kwargs']['resize']['p']),
            transforms.RandomHorizontalFlip(p=p['augmentation_kwargs']['horizontal_flip']['p']),
            transforms.RandomVerticalFlip(p=p['augmentation_kwargs']['vertical_flip']['p']),
            transforms.RandomApply([Rotate90deg([-90, 90, 180, -180])], p['augmentation_kwargs']['rotate90']['p']),
            transforms.RandomApply([
                transforms.ColorJitter(**p['augmentation_kwargs']['color_jitter'])
            ], p=p['augmentation_kwargs']['color_jitter_random_apply']['p']),
            transforms.RandomGrayscale(**p['augmentation_kwargs']['random_grayscale']),
            transforms.ToTensor(),
            #transforms.Normalize(**p['augmentation_kwargs']['normalize'])
        ])

    elif p['augmentation_strategy'] == 'simclr':
        # Augmentation strategy from the SimCLR paper
        return transforms.Compose([
            transforms.RandomResizedCrop(**p['augmentation_kwargs']['random_resized_crop']),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(**p['augmentation_kwargs']['color_jitter'])
            ], p=p['augmentation_kwargs']['color_jitter_random_apply']['p']),
            transforms.RandomGrayscale(**p['augmentation_kwargs']['random_grayscale']),
            transforms.ToTensor(),
            #transforms.Normalize(**p['augmentation_kwargs']['normalize'])
        ])
    
    elif p['augmentation_strategy'] == 'scan_og_paper':
        # Augmentation strategy from our paper 
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(p['augmentation_kwargs']['crop_size']),
            Augment(p['augmentation_kwargs']['num_strong_augs']),
            transforms.ToTensor(),
            #transforms.Normalize(**p['augmentation_kwargs']['normalize']),
            Cutout(
                n_holes = p['augmentation_kwargs']['cutout_kwargs']['n_holes'],
                length = p['augmentation_kwargs']['cutout_kwargs']['length'],
                random = p['augmentation_kwargs']['cutout_kwargs']['random'])])
    
    elif p['augmentation_strategy'] == 'none':
        return transforms.Compose([
               transforms.ToTensor(), 
               #transforms.Normalize(**p['transformation_kwargs']['normalize'])
               ])
    
    else:
        raise ValueError('Invalid augmentation strategy {}'.format(p['augmentation_strategy']))


def get_val_transformations(p):
    return transforms.Compose([
            transforms.ToTensor(), 
            #transforms.Normalize(**p['transformation_kwargs']['normalize'])
            ])


def get_optimizer(p, model, cluster_head_only=False):
    if cluster_head_only: # Only weights in the cluster head will be updated 
        for name, param in model.named_parameters():
                if 'cluster_head' in name:
                    param.requires_grad = True 
                else:
                    param.requires_grad = False 
        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        assert(len(params) == 2 * p['num_heads'])

    else:
        params = model.parameters()
                

    if p['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params, **p['optimizer_kwargs'])

    elif p['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params, **p['optimizer_kwargs'])
    
    else:
        raise ValueError('Invalid optimizer {}'.format(p['optimizer']))

    return optimizer


def adjust_learning_rate(p, optimizer, epoch):
    lr = p['optimizer_kwargs']['lr']
    
    if p['scheduler'] == 'cosine':
        eta_min = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / p['epochs'])) / 2
         
    elif p['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(p['scheduler_kwargs']['lr_decay_epochs']))
        if steps > 0:
            lr = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** steps)

    elif p['scheduler'] == 'constant':
        lr = lr

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr