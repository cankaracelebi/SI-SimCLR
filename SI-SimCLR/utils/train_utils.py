"""
Original Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
Modifications by Elisa Messori for SI-SimCLR development
"""
import torch
import numpy as np
import pdb
import time
from utils.utils import AverageMeter, ProgressMeter, compute_entropy, get_truncated_model
import os
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils.utils import average_euclidean_distance


def simclr_train(train_loader, model, criterion, optimizer, epoch, batch_size, total_epochs, base_directory):
    """ 
    Train according to the scheme from SimCLR
    https://arxiv.org/abs/2002.05709
    Modified version by Elisa Messori to return loss
    """
    #pdist = torch.nn.PairwiseDistance()
    #cos_sim = torch.nn.CosineSimilarity()
    losses = AverageMeter('Loss (training)', ':.4e')
    progress = ProgressMeter(len(train_loader),
        [losses],
        prefix="Epoch: [{}]".format(epoch))
    model.train()
    device = next(model.parameters()).device
    #indices = []
    #embeddings = []
    if device.type == 'cuda':
        start = torch.cuda.Event(enable_timing=True) # Start/end are used to compute epoch time on GPU
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    else:
        start_time = time.perf_counter()

    for i, batch in enumerate(train_loader):
        #avg_dist_batch = []
        images = batch['image']
        images_augmented = batch['image_augmented']
        b, c, h, w = images.size()
        input_ = torch.cat([images.unsqueeze(1), images_augmented.unsqueeze(1)], dim=1)
        input_ = input_.view(-1, c, h, w) 
        input_ = input_.to(device, non_blocking=(device.type == 'cuda'))
        # targets = batch['target'].cuda(non_blocking=True)

        output = model(input_).view(b, 2, -1)
        loss = criterion(output)

        losses.update(loss.item())
        
        # HERE I CAN STORE THE AVERAGE LOSS OVER THE BATCHES OF A CERTAIN EPOCH IN ORDER TO PLOT THEM LATER
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        if i % 150 == 0:
            progress.display(i)

        #pairs = torch.unbind(output, dim=1)
        #avg_dist_batch = pdist(pairs[0], pairs[1])
        #avg_dist_batch_cos = 1 - cos_sim(pairs[0], pairs[1])

    #avg_dist_epoch = float(avg_dist_batch.mean())
    #print('Avg dist epoch (euclidean)')
    #print(avg_dist_epoch)

    #avg_dist_epoch_cos = float(avg_dist_batch_cos.mean())
    #print('Avg dist epoch (cosine)')
    #print(avg_dist_epoch_cos)
    if device.type == 'cuda':
        end.record()
        torch.cuda.synchronize()
        time_interval = start.elapsed_time(end)/1e3
    else:
        time_interval = time.perf_counter() - start_time
    print('Time of training for epoch %3d: %5.4f seconds' % (epoch, time_interval))

    return losses.avg #, avg_dist_epoch, avg_dist_epoch_cos

def simclr_validation(val_loader, model, criterion, optimizer, epoch):
    """ 
    For plotting loss on validation set
    """
    pdist = torch.nn.PairwiseDistance()
    cos_sim = torch.nn.CosineSimilarity()
    losses = AverageMeter('Loss (validation)', ':.4e')
    progress = ProgressMeter(len(val_loader),
        [losses],
        prefix="Epoch: [{}]".format(epoch))
    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            images = batch['image']
            images_augmented = batch['image_augmented']
            b, c, h, w = images.size()
            input_ = torch.cat([images.unsqueeze(1), images_augmented.unsqueeze(1)], dim=1)
            input_ = input_.view(-1, c, h, w) 
            input_ = input_.to(device, non_blocking=(device.type == 'cuda'))
            # targets = batch['target'].cuda(non_blocking=True)

        output = model(input_).view(b, 2, -1)
        loss = criterion(output)
        losses.update(loss.item())

        if i % 150 == 0:
            progress.display(i)
        
        pairs = torch.unbind(output, dim=1)
        avg_dist_batch = pdist(pairs[0], pairs[1])
        avg_dist_batch_cos = 1 - cos_sim(pairs[0], pairs[1])

    avg_dist_epoch = float(avg_dist_batch.mean())
    print('Avg dist epoch (euclidean)')
    print(avg_dist_epoch)

    avg_dist_epoch_cos = float(avg_dist_batch_cos.mean())
    print('Avg dist epoch (cosine)')
    print(avg_dist_epoch_cos)

    return losses.avg, avg_dist_epoch, avg_dist_epoch_cos


# Original Version
#def simclr_train(train_loader, model, criterion, optimizer, epoch):
#    """ 
#    Train according to the scheme from SimCLR
#    https://arxiv.org/abs/2002.05709
#    """
#    losses = AverageMeter('Loss', ':.4e')
#    progress = ProgressMeter(len(train_loader),
#        [losses],
#        prefix="Epoch: [{}]".format(epoch))
#
#    model.train()
#
#    for i, batch in enumerate(train_loader):
#        images = batch['image']
#        images_augmented = batch['image_augmented']
#        b, c, h, w = images.size()
#        input_ = torch.cat([images.unsqueeze(1), images_augmented.unsqueeze(1)], dim=1)
#        input_ = input_.view(-1, c, h, w) 
#        input_ = input_.cuda(non_blocking=True)
#        targets = batch['target'].cuda(non_blocking=True)
#
#        output = model(input_).view(b, 2, -1)
#
#        #pdb.set_trace()
#
#        loss = criterion(output)
#        losses.update(loss.item())
#
#        optimizer.zero_grad()
#        loss.backward()
#        optimizer.step()
#
#        if i % 25 == 0:
#            progress.display(i)
#
