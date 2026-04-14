"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

Modifications by Elisa Messori: added collate option for pandas dataframe
"""
import torch
import numpy as np
import collections.abc as abc
from PIL import Image
import pandas as pd
#from torch._six import string_classes


""" Custom collate function """
def collate_custom(batch):

    if isinstance(batch[0], (int, np.int64)):
        return torch.LongTensor(batch)

    elif isinstance(batch[0], (float, np.float32, np.float64)):
        return torch.FloatTensor(batch)
    
    elif isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)
    
    elif isinstance(batch[0], np.ndarray):
        return np.stack(batch, 0)

    elif isinstance(batch[0], dict):
        batch_modified = {key: collate_custom([d[key] for d in batch]) for key in batch[0] if 'idx' not in key}
        return batch_modified

    elif isinstance(batch[0], (list, tuple)):
        transposed = zip(*batch)
        return [collate_custom(samples) for samples in transposed]

    elif isinstance(batch[0], float):
        return torch.FloatTensor(batch)
    
    elif isinstance(batch[0], pd.DataFrame):
        # We do not use the dataframe for training but only for plotting and downstream analysis
        return batch

    elif isinstance(batch[0], str):
        return batch

    #elif isinstance(batch[0], abc.Mapping):
    #    batch_modified = {key: collate_custom([d[key] for d in batch]) for key in batch[0] if key.find('idx') < 0}
    #    return batch_modified
#
    #elif isinstance(batch[0], abc.Sequence):
    #    transposed = zip(*batch)
    #    return [collate_custom(samples) for samples in transposed]

    raise TypeError(('Type is {}'.format(type(batch[0]))))
