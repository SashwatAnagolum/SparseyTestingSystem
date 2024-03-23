# -*- coding: utf-8 -*-

"""
Dataset: file holding the dataset class.
"""


import os

from typing import Any

import torch
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self):
        """
        """
        super().__init__()
