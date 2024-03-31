# -*- coding: utf-8 -*-

"""
Built In Dataset: file holding the built in dataset class.
"""


import os

from typing import Any

import numpy as np

from torchvision.io import read_image
from torchvision import datasets
from torch.utils.data import Dataset as TorchDataset
from torchvision.transforms import v2

from sparseypy.access_objects.datasets.dataset import Dataset


class BuiltInDataset(Dataset):
    """
    
    """
    def __init__(self, name: str, root: str,
                 download: bool, transform: str):
        """
        
        """
        self.dataset_name = name
        self.dataset_folder = root
        self.transform = getattr(v2, transform)()

        self.wrapped_dataset: TorchDataset = getattr(
            datasets, name
        )(root=root, download=download, transform=self.transform)


    def __getitem__(self, index) -> Any:
        return self.wrapped_dataset.__getitem__(index)


    def __len__(self) -> int:
        return self.wrapped_dataset.__len__()
