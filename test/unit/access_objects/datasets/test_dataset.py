import pytest
from sparseypy.access_objects.datasets import ImageDataset  # Adjust the import according to your file structure
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class TestImageDataset:
    @pytest.fixture
    def mnist_dataset(self):
        # Initialize your dataset here
        # Ensure that set withe path to sample_mnist_dataset on
        return ImageDataset(data_dir='C:\\Users\\toddk\\Downloads\\SparseyTestingSystem\\demo\\sample_mnist_dataset', image_format='.png')

    def test_getitem(self, mnist_dataset):
        # Test if __getitem__ returns an image and its label correctly
        image, label = mnist_dataset[0]
        assert image.dim() == 3  # CxHxW format for an image
        assert type(label) == int
        assert 0 <= label <= 9  # MNIST labels are from 0 to 9

    def test_dataloader_integration(self, mnist_dataset):
        # Test if the dataset is compatible with DataLoader
        dataloader = DataLoader(mnist_dataset, batch_size=10, shuffle=True)
        images, labels = next(iter(dataloader))
        assert images.shape[0] == 10  # Batch size
        assert images.dim() == 4  # CxHxW format for images
        assert len(labels) == 10  # Batch size
        assert all(0 <= label <= 9 for label in labels)  # MNIST labels are from 0 to 9
"""
03/09/2024 - Andy Klawa
PS C:\Users\toddk\Downloads\SparseyTestingSystem> pytest test\unit\access_objects\datasets\test_dataset.py
============================================== test session starts ===============================================
platform win32 -- Python 3.11.6, pytest-7.4.3, pluggy-1.3.0
rootdir: C:\Users\toddk\Downloads\SparseyTestingSystem
collected 2 items

test\unit\access_objects\datasets\test_dataset.py ..                                                        [100%]

=============================================== 2 passed in 7.14s ================================================ 
"""