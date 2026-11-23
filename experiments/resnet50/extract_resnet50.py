"""
Frozen ImageNet-pretrained ResNet50 embeddings for the astrocyte crops.

Saves embeddings/<exp_code>/latent_embeddings_train.pickle in the same layout as
SI-SimCLR/simclr.py, so the downstream analyses can load it by experiment code.

Run from the repository root:
    python experiments/resnet50/extract_resnet50.py \
        --config_env configs/env.yml \
        --config_exp configs/resnet50_imagenet/simclr_acs.yml
"""
import os, sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path[:0] = [REPO_ROOT, os.path.join(REPO_ROOT, 'SI-SimCLR')]

import argparse
from collections import OrderedDict
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tempfile                                     # These two lines are supposed to 'solve' the matplotlib warning
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
from tqdm import tqdm
from utils.utils import experiments_summary
from utils.config import create_config
from utils.common_config import get_train_dataset, get_val_dataloader
from utils.inference import get_latent_embeddings, PCA_latent_embeddings


# Parser
parser = argparse.ArgumentParser(description='Frozen ResNet50 embeddings')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
parser.add_argument('--overwrite', action='store_true',
                    help='Recompute embeddings even if they already exist')
args = parser.parse_args()

IMAGENET_MEAN = [0.485, 0.456, 0.406] # apply same transformation like it got trained
IMAGENET_STD = [0.229, 0.224, 0.225]


BACKBONES = {'resnet18': (resnet18, ResNet18_Weights.IMAGENET1K_V1),   # 512-d features
             'resnet50': (resnet50, ResNet50_Weights.IMAGENET1K_V1)}   # 2048-d features


def get_frozen_resnet(name):
    if name == 'dino_vits16':
        # ViT-S/16 pretrained on ImageNet with DINO (self-supervised); forward returns the 384-d CLS token
       
        is_utils = lambda k: k == 'utils' or k.startswith('utils.')
        ours = {k: sys.modules.pop(k) for k in list(sys.modules) if is_utils(k)}
        try:
            backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', trust_repo=True)
        finally:
            for k in [k for k in sys.modules if is_utils(k)]:
                del sys.modules[k]
            sys.modules.update(ours)
    else:
        builder, weights = BACKBONES[name]
        backbone = builder(weights=weights)
        backbone.fc = nn.Identity()  # output = pooled features
    for param in backbone.parameters():
        param.requires_grad = False
  
    return nn.Sequential(OrderedDict([('backbone', backbone), ('head', nn.Identity())]))


def main():

    # Retrieve config file
    p = create_config(args.config_env, args.config_exp)
    print(p)

    print('')
    print(f'LAUNCHING EXPERIMENT {p["exp_code"]}')
    print('')
    exp_summary = experiments_summary(p)

    # Model
    print('Retrieve model')
    model = get_frozen_resnet(p['backbone'])  # 'backbone' key of the experiment config
    print(f"Backbone: {p['backbone']}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device:\t{device}')
    print('Model parameters: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    model = model.to(device).eval()

    # Dataset: plain (no pairing wrapper), deterministic ImageNet preprocessing
    print('Retrieve dataset')
    eval_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    print('Transforms:', eval_transforms)
    dataset = get_train_dataset(p, eval_transforms)
    print(f'Dataset contains {len(dataset)} images')

    # shuffle=False, drop_last=False -> rows follow the annotation CSV order
    dataloader = get_val_dataloader(p, dataset)
    emb_path = os.path.join(p['target_dir'], 'latent_embeddings_train.pickle')
    if args.overwrite or not os.path.exists(emb_path):
        progress = tqdm(dataloader, desc=f"{p['backbone']} embeddings", unit='batch')
        get_latent_embeddings(model, progress, p['target_dir'], p['batch_size'])
        print('Latent embeddings saved')
    else:
        print(f'{emb_path} already exists, skipping extraction (use --overwrite to recompute)')

    exp_summary.loc[exp_summary['exp codes'] == p['exp_code'], 'trained'] = 1
    exp_summary.to_csv(os.path.join(p['results_root_dir'], 'exp_summary.csv'), index=False)

    TARGET_DIR = p['target_dir']
    if args.overwrite or not os.path.exists(os.path.join(TARGET_DIR, 'PCA_embeddings_train.pickle')):
        PCA_latent_embeddings(TARGET_DIR, False)

    if not os.path.exists(os.path.join(TARGET_DIR, 'plots')):
        os.makedirs(os.path.join(TARGET_DIR, 'plots'))

if __name__ == '__main__':
    main()
