# -*- coding: utf-8 -*-

"""
Dataset: file holding the dataset class.
"""


import os

from typing import Any

import torch
from torch.utils import data


class Dataset(data.Dataset):
    """
    Parent dataset class that wraps around a torch.utils.data.Dataset.
    """
    def __init__(self):
        """
        Initialize the Dataset.
        """
        super().__init__()
