# -*- coding: utf-8 -*-

"""
IMage Dataset: file holding the image dataset class.
"""


import os
import torch

import numpy as np

from typing import Any

from torchvision.io import read_image

from sparseypy.access_objects.datasets.dataset import Dataset


class ImageDataset(Dataset):
    """
    A dataset class that contains images.
    Attributes:
        data_folder (str): The directory where the images are stored.
        subfolders (list[str]): The subfolders in the data folder.
        subfolder_images (list[list[str]]): The images in each subfolder.
        subfolder_image_counts (list[int]): The number of images in each subfolder.
        total_images (int): The total number of images in the dataset.
    """
    def __init__(self, data_dir: str, image_format: str):
        """
        Initialize the ImageDataset.
        Args:
            data_dir (str): The directory where the images are stored.
            image_format (str): The format of the images.
        """
        super().__init__()

        self.data_folder = data_dir
        self.subfolders = []
        self.subfolder_images = []
        self.subfolder_image_counts = [0]

        for (
            folder, _, files
        ) in os.walk(data_dir):
            if folder == self.data_folder:
                continue

            self.subfolders.append(folder)
            self.subfolder_images.append(
                [i for i in files if os.path.splitext(i)[1] == image_format]
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
        Get an item from the dataset.
        Args:
            index (int): The index of the item to get.
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

        image = read_image(image_path)

        return image, image_subfolder
    

    def __len__(self):
        """
        Get the length of the dataset.
        """
        return self.total_images
