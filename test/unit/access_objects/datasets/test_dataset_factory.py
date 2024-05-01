import pytest
from sparseypy.access_objects.datasets.dataset_factory import DatasetFactory
from torch.utils.data import Dataset

from sparseypy.access_objects.datasets.image_dataset import ImageDataset

class TestDatasetFactory:
    """
    Unit tests for the DatasetFactory class.
    """

    def test_valid_dataset_configuration(self):
        """
        TC-12-01: Test the DatasetFactory's ability to process a valid dataset configuration 
        and return a corresponding Dataset object.
        """
        # Assuming we have a dataset class named 'ExampleDataset' in the datasets module
        valid_dataset_config = {
            'data_dir': './demo/sample_mnist_dataset',
            'image_format': '.png'   
        }
        # This example assumes there's a corresponding 'ExampleDataset' in the datasets module
        dataset = DatasetFactory.create_dataset('image', **valid_dataset_config)
        
        assert isinstance(dataset, ImageDataset), "The returned object is not an instance of Dataset"

    def test_invalid_dataset_configuration(self):
        """
        TC-12-02: Test the robustness of the DatasetFactory in identifying and rejecting
        invalid or incomplete dataset configurations.
        """
        # Assuming the dataset type 'unknown' does not exist
        # Assuming we have a dataset class named 'ExampleDataset' in the datasets module
        invalid_dataset_config = {
            'data_dir': './demo/sample_mnist_dataset'
        }
        with pytest.raises(TypeError) as excinfo:
            DatasetFactory.create_dataset('image', **invalid_dataset_config)
        
        assert "ImageDataset.__init__() missing 1 required positional argument: 'image_format'" in str(excinfo.value), "Error message for invalid dataset type is not correct or not raised"

