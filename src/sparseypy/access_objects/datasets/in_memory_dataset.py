# -*- coding: utf-8 -*-
"""
In Memory Dataset: wrapper for datasets
"""


import functools

from sparseypy.access_objects.datasets.dataset import Dataset


class InMemoryDataset(Dataset):
    """
    A dataset wrapper class that pre-fetches
    data samples from the dataset and stores them in memory
    to increase throughput.

    Attributes:
        dataset (Dataset): The original dataset to be wrapped.
    """
    def __init__(self, dataset: Dataset, fetch_lazily: bool = True):
        """
        Initialize the InMemoryDataset.

        Args:
            dataset (Dataset): The dataset to be preprocessed.
            fetch_lazily (bool): whether to fetch the items 
                lazily (when requested for the first time)
                or eagerly (all samples fetched at initialization).
        """
        self.dataset = dataset

        if not fetch_lazily:
            for i in range(self.__len__()):
                self.__getitem__(i)


    @functools.lru_cache(maxsize=None)
    def __getitem__(self, idx):
        """
        Get item by index from disk if not sampled before,
        otherwise return from the cache.

        Args:
            idx (int): Index of the data.

        Returns:
            Data and its label.
        """
        return self.dataset.__getitem__(idx)


    def __len__(self):
        """
        Return the length of the dataset.
        Returns:
            Length of the dataset.
        """
        return len(self.dataset)
