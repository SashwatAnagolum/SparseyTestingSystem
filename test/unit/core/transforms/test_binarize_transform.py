import pytest
import torch

from sparseypy.core.transforms.binarize_transform import BinarizeTransform  # Update the import path as needed

class TestBinarizeTransform:
    def test_binarization_accuracy(self):
        """
        Test the accuracy of the binarization transform in processing data
        and converting it to a binary format.

        Test Case ID: TC-06-02
        """
        # Create a test tensor with incremental values
        test_tensor = torch.torch.tensor([[[0.1],
            [0.2],
            [0.3],
            [0.4],
            [0.45],
            [0.49],
            [0.51],
            [0.9],
            [1.],
            [0.99]]])
        transform = BinarizeTransform(binarize_threshold=0.5)
        
        # Apply the binarize transform
        transformed_tensor = transform.forward(test_tensor)
        print(transformed_tensor)
        # Check that the values are binarized correctly
        expected_output = torch.tensor([[[0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [1.],
            [1.],
            [1.],
            [1.]]])
        assert torch.equal(transformed_tensor, expected_output), "Binarization did not work as expected."
