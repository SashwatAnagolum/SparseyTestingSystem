# -*- coding: utf-8 -*-

"""
Sparsey Dataset: file holding the Sparsey dataset class to read data from Sparsey binary RAW files.
"""
from typing import Any

import os

import numpy as np
import torch

from sparseypy.access_objects.datasets.dataset import Dataset


class SparseyDataset(Dataset):
    """
    Dataset adapter class to read existing binary Sparsey datasets.

    Requires the width and height of the dataset images be provided 
    in order to correctly interpret the raw files.

    Note this adapter currently does NOT support reading 
    multiple items from the same binary file.
    """
    def __init__(self, data_dir: str, width: int, height: int):
        """
        Constructor for the SparseyDataset.

        Args:
            data_dir (str): the path to the folder containing the dataset
            width (int): the width in pixels of the dataset input images
            height (int): the height in pixels of the dataset input images
        """
        super().__init__()

        self.data_folder = data_dir
        self.subfolders = []
        self.subfolder_images = []
        self.subfolder_image_counts = [0]

        self.width = width
        self.height = height

        for (
            folder, _, files
        ) in os.walk(data_dir):
            if folder == self.data_folder:
                continue

            self.subfolders.append(folder)
            self.subfolder_images.append(
                # Sparsey dataset input files are in .RAW format
                [i for i in files if os.path.splitext(i)[1] == ".raw"]
            )

            self.subfolder_image_counts.append(
                len(self.subfolder_images[-1])
            )

        self.subfolder_image_counts = np.cumsum(
            self.subfolder_image_counts
        )

        self.total_images = self.subfolder_image_counts[-1]


    def __getitem__(self, index) -> Any:
        """
        Retrieves the item at the indicated index from the SparseyDataset.
        """
        image_subfolder = np.argwhere(
            self.subfolder_image_counts <= index
        )[-1].item()

        image_subfolder_index = index - self.subfolder_image_counts[
            image_subfolder
        ]

        image_path = os.path.join(
            self.subfolders[image_subfolder],
            self.subfolder_images[image_subfolder][image_subfolder_index]
        )

        # Sparsey dataset entries are 1 byte per value
        raw_data = np.fromfile(image_path, dtype='uint8')
        # Sparsey dataset is stored as width x height
        reshaped_data = np.reshape(raw_data, (1, self.height, self.width))
        # convert to tensor and return
        image = torch.from_numpy(reshaped_data)

        return image, image_subfolder

    def __len__(self):
        """
        Returns the length of the SparseyDataset.
        """
        return self.total_images
