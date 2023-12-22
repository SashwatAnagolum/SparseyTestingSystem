# -*- coding: utf-8 -*-

"""
IMage Dataset: file holding the image dataset class.
"""


import os
import torch

import numpy as np

from typing import Any

from torchvision.io import read_image

from sparsepy.access_objects.datasets.dataset import Dataset


class ImageDataset(Dataset):
    def __init__(self, data_dir: str, image_format: str):
        """
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
        return self.total_images
