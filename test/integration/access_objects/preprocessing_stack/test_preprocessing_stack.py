import pytest
import torch
from torchvision.transforms.v2._color import Grayscale, RandomPosterize
from sparseypy.access_objects.datasets import ImageDataset
from sparseypy.access_objects.datasets import PreprocessedDataset
from sparseypy.core.transforms.transform_factory import TransformFactory
from sparseypy.access_objects.preprocessing_stack.preprocessing_stack import PreprocessingStack

# Setup for tests
@pytest.fixture
def mnist_image_dataset():
    # Adjust the `data_dir` path to where your MNIST dataset is located
    return ImageDataset(data_dir=".\\demo\\sample_mnist_dataset", image_format=".png")


@pytest.fixture
def simple_preprocessing_stack():
    # Define a simple preprocessing configuration for demonstration
    transform_configs = {
        'transform_list': [
            {'name': 'grayscale', 'params': {'num_output_channels': 1}},
            {'name': 'random_posterize', 'params': { 'p': 1.0, 'bits': 1,}} 
        ]
    }

    return PreprocessingStack(transform_configs)


@pytest.mark.usefixtures("mnist_image_dataset", "simple_preprocessing_stack")
def test_transform_order(mnist_image_dataset, simple_preprocessing_stack):
    """
    Test case ID: TC-14-01
    """
    preprocessed_dataset = PreprocessedDataset(dataset=mnist_image_dataset, preprocessing_stack=simple_preprocessing_stack)

    assert isinstance(preprocessed_dataset.preprocessing_stack.transform_list[0], Grayscale), "First transform should be ToTensor"
    assert isinstance(preprocessed_dataset.preprocessing_stack.transform_list[1], RandomPosterize), "Second transform should be Normalize"