import pytest
import torch
import time

from sparseypy.access_objects.datasets import ImageDataset
from sparseypy.access_objects.datasets import PreprocessedDataset
from sparseypy.core.transforms.transform_factory import TransformFactory
from sparseypy.access_objects.preprocessing_stack.preprocessing_stack import PreprocessingStack

class BasicTransformFactory:
    """
    TC-03-02: Tests the PreprocessedDataset class's ability to correctly and efficiently wrap an ImageDataset with preprocessing applied.
    """

    @staticmethod
    def create_transform(name, **params):
        if name == 'ToTensor':
            return torch.nn.Identity()
        elif name == 'Normalize':
            return torch.nn.Identity() 
        else:
            raise ValueError(f"Unknown transform: {name}")

TransformFactory.create_transform = BasicTransformFactory.create_transform

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
            {'name': 'ToTensor', 'params': {}},
            {'name': 'Normalize', 'params': {'mean': 0.5, 'std': 0.5}} 
        ]
    }
    return PreprocessingStack(transform_configs)

@pytest.fixture
def preprocessed_mnist_dataset(mnist_image_dataset, simple_preprocessing_stack):
    return PreprocessedDataset(dataset=mnist_image_dataset, preprocessing_stack=simple_preprocessing_stack)

# Tests
def test_preprocessed_dataset_length(preprocessed_mnist_dataset, mnist_image_dataset):
    assert len(preprocessed_mnist_dataset) == len(mnist_image_dataset), "Preprocessed dataset length does not match the original dataset."

def test_preprocessing_applied(preprocessed_mnist_dataset):
    preprocessed_data, label = preprocessed_mnist_dataset[0]
    assert isinstance(preprocessed_data, torch.Tensor), "Preprocessed data is not a torch.Tensor"

def test_preprocessing_cache(preprocessed_mnist_dataset):
    # First retrieval - likely not cached yet, so it might take longer
    start_time = time.perf_counter()
    first_retrieval = preprocessed_mnist_dataset[0]
    first_duration = time.perf_counter() - start_time

    # Second retrieval - should be faster if caching is effective
    start_time = time.perf_counter()
    second_retrieval = preprocessed_mnist_dataset[0]
    second_duration = time.perf_counter() - start_time

    # Ensure the data and label are consistent across retrievals
    assert first_retrieval[0].equal(second_retrieval[0]), "Preprocessed data does not match across retrievals."
    assert first_retrieval[1] == second_retrieval[1], "Labels do not match across retrievals."

    # Check if the second retrieval was indeed faster, indicating effective caching
    assert second_duration < first_duration, "Second retrieval was not faster than the first; caching may not be effective."