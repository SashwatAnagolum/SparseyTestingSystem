"""
Test Image Dataset: test cases for the ImageDataset class.
"""

import pytest
import torch
from torch.utils.data import DataLoader
import os

from sparseypy.access_objects.datasets import ImageDataset

class TestImageDataset:
    """
    TestImageDataset: a class holding a collection
        of tests focused on the ImageDataset class.
    """

    def test_image_dataset_integration_with_dataloader(self) -> None:
        """
        TC-03-01: Tests whether the ImageDataset integrates correctly with PyTorch DataLoaders and returns
        data in the expected format.
        """
        # Assuming a hypothetical directory structure and image format
        data_directory = '.\demo\sample_mnist_dataset'
        image_format = '.png'
        batch_size = 10

        # Instantiate the ImageDataset
        dataset = ImageDataset(data_directory, image_format)

        # Using DataLoader to handle the dataset
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Fetch one batch of data
        images, labels = next(iter(dataloader))

        # Check if the DataLoader provides data in the correct batch size
        assert len(images) == batch_size, "Batch size does not match expected"

        # Check the shape of the images (assuming the MNIST images are 28x28 and grayscale)
        assert images.shape == (batch_size, 1, 28, 28), "Image shape does not match expected format"

        # Check the type of images and labels to ensure they are tensors
        assert isinstance(images, torch.Tensor), "Images are not returned as torch.Tensor"
        assert isinstance(labels, torch.Tensor), "Labels are not returned as torch.Tensor"

        # Optionally, check the uniqueness and type of labels if relevant to the test
        # This part depends on the specifics of the dataset and what labels represent
        assert images.dtype == torch.uint8, "Images are not in the expected uint8 format"
        assert labels.dtype == torch.int64, "Labels are not in the expected int64 format"

# Additional tests can be added to the TestImageDataset class to cover more aspects like error handling,
# performance, or other specific functionalities of the ImageDataset class.
