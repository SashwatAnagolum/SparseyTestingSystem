import pytest
import torch
from torchvision.transforms.v2 import Resize
from sparseypy.core.transforms.skeletonize_transform import SkeletonizeTransform
from sparseypy.core.transforms.transform_factory import TransformFactory 

def test_invalid_transform_creation():
    """
    Test TransformFactory with invalid transform parameters.

    Test Case ID: TC-08-02
    """
    # Attempt to create a transform with an invalid name
    with pytest.raises(ValueError) as exc_info:
        TransformFactory.create_transform('non_existent_transform')


def test_valid_system_transform():
    """
    Test TransformFactory with valid system transform parameters.

    Test Case ID: TC-08-01
    """
    # Create a ToTensor transform
    transform = TransformFactory.create_transform('skeletonize')
    
    # Check if the transform is an instance of  SkeletonizeTransform
    assert isinstance(transform, SkeletonizeTransform), "Failed to create a  SkeletonizeTransform transform."


def test_valid_pytorch_transform():
    """
    Test TransformFactory with valid Pytorch transform parameters.

    Test Case ID: TC-08-03
    """
    config = {
        'name': 'resize',
        'params': {
            'size': [8, 8],
            'antialias': True
        }
    }

    # Create a ToTensor transform
    transform = TransformFactory.create_transform(config['name'], **config['params'])
    
    # Check if the transform is an instance of  ToTensor
    assert isinstance(transform, Resize), "Failed to create a Resize transform."

