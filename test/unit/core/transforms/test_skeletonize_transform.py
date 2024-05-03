import pytest
import torch
import numpy as np

from skimage import data
from sparseypy.core.transforms.skeletonize_transform import SkeletonizeTransform  # Update the import path as needed

class TestSkeletonizeTransform:
    def test_skeletonization_processing(self):
        """
        Test whether the skeletonization transform processes and outputs
        data correctly, performing skeletonization as expected on a simple geometric shape.

        Test Case ID: TC-06-01
        """
        # Create a simple binary image tensor with a filled rectangle
        test_image = torch.zeros((1, 10, 10))
        test_image[:, 3:7, 4:6] = 1  # A vertical rectangle
        
        transform = SkeletonizeTransform(sigma=0)  # Sigma is set to 0 since we're not using edge detection here
        
        # Apply the skeletonization transform
        transformed_tensor = transform.forward(test_image)
        
        # Known expected output: skeleton of the rectangle should be a line
        expected_output = torch.tensor([[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 1., 1., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]], dtype=torch.float64)
        
        # Assert against the exact output
        assert torch.equal(transformed_tensor, expected_output), "Skeletonization did not produce the expected output"