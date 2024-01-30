# -*- coding: utf-8 -*-

"""
Preprocessed Dataset: wrapper for datasets
"""
import functools
from sparsepy.access_objects.datasets.dataset import Dataset
from sparsepy.access_objects.preprocessing_stack.preprocessing_stack import PreprocessingStack

class PreprocessedDataset(Dataset):
    """
    A dataset wrapper class that applies preprocessing to another dataset and caches the results using lru_cache.

    Attributes:
        dataset (Dataset): The original dataset to be preprocessed.
        preprocessing_stack (PreprocessingStack): The preprocessing operations to be applied.
    """

    def __init__(self, dataset: Dataset, preprocessing_stack: PreprocessingStack):
        """
        Initialize the PreprocessedDataset.

        Args:
            dataset (Dataset): The dataset to be preprocessed.
            preprocessing_stack (PreprocessingStack): Stack of preprocessing steps to apply.
        """
        self.dataset = dataset
        self.preprocessing_stack = preprocessing_stack

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, idx):
        """
        Get item by index, applying preprocessing and caching the result.

        Args:
            idx (int): Index of the data.

        Returns:
            Preprocessed data and its label.
        """
        # Retrieve data and label from the original dataset
        data, label = self.dataset[idx]

        # Apply preprocessing steps
        preprocessed_data = self.preprocessing_stack(data)

        return preprocessed_data, label

    def __len__(self):
        """
        Return the length of the dataset.

        Returns:
            Length of the dataset.
        """
        return len(self.dataset)
