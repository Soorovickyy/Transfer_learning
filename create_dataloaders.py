from torch.utils.data import DataLoader
from torchvision import datasets
import os

import config


def get_dataloader(rootDir, transforms, batchSize,  shuffle=True):
    ds = datasets.ImageFolder(root=rootDir,
                              transform=transforms)
    loader = DataLoader(ds, batch_size=batchSize, shuffle=shuffle)
    return ds, loader