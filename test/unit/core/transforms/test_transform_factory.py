import pytest
import torch
from torchvision.transforms import ToTensor
from sparseypy.core.transforms.skeletonize_transform import SkeletonizeTransform
from sparseypy.core.transforms.transform_factory import TransformFactory 

def test_create_to_tensor_transform():
    # Create a ToTensor transform
    transform = TransformFactory.create_transform('skeletonize')
    
    # Check if the transform is an instance of  SkeletonizeTransform
    assert isinstance(transform, SkeletonizeTransform), "Failed to create a  SkeletonizeTransform transform."

def test_invalid_transform_creation():
    # Attempt to create a transform with an invalid name
    with pytest.raises(ValueError) as exc_info:
        TransformFactory.create_transform('non_existent_transform')

