# -*- coding: utf-8 -*-
"""
Preprocessed Dataset: wrapper for datasets
"""
import os
import pickle
import functools
from sparsepy.access_objects.datasets.dataset import Dataset
from sparsepy.access_objects.preprocessing_stack.preprocessing_stack import PreprocessingStack

class PreprocessedDataset(Dataset):
    """
    A dataset wrapper class that applies preprocessing to another dataset and caches the results.
    Attributes:
        dataset (Dataset): The original dataset to be preprocessed.
        preprocessed_dir (str): Directory where preprocessed data is stored.
        preprocessing_stack (PreprocessingStack): The preprocessing operations to be applied.
        preprocessed_flags (list[bool]): A boolean list indicating whether an item has been preprocessed.
    """

    def __init__(self, dataset: Dataset, preprocessing_stack: PreprocessingStack, preprocessed_dir: str = "datasets/preprocessed_datasets"):
        """
        Initialize the PreprocessedDataset.
        Args:
            dataset (Dataset): The dataset to be preprocessed.
            preprocessed_dir (str): Directory to store preprocessed data.
            preprocessing_stack (PreprocessingStack): Stack of preprocessing steps to apply.
        """
        self.dataset = dataset
        self.preprocessed_dir = preprocessed_dir
        self.preprocessing_stack = preprocessing_stack
        self.preprocessed_flags = [False] * len(dataset)  # Initialize all flags as False

        # Create the directory for preprocessed data if it does not exist
        if not os.path.exists(self.preprocessed_dir):
            os.makedirs(self.preprocessed_dir)

    def _preprocess_and_save(self, idx):
        """
        Preprocess and save data for a given index.
        Args:
            idx (int): Index of the data in the dataset.
        Returns:
            Tuple containing preprocessed data and its label.
        """
        # Retrieve data and label from the original dataset
        data, label = self.dataset[idx]

        # Apply preprocessing steps
        preprocessed_data = self.preprocessing_stack(data)

        # Path where the preprocessed data will be saved
        preprocessed_path = os.path.join(self.preprocessed_dir, f'{idx}.pkl')

        # Save the preprocessed data and label
        with open(preprocessed_path, 'wb') as f:
            pickle.dump((preprocessed_data, label), f)

        # Mark this item as preprocessed in the boolean array
        self.preprocessed_flags[idx] = True

        return preprocessed_data, label
    
    @functools.lru_cache(maxsize=None)
    def __getitem__(self, idx):
        """
        Get item by index, applying preprocessing if necessary.
        Args:
            idx (int): Index of the data.
        Returns:
            Preprocessed data and its label.
        """
        # Path to the preprocessed file for this index
        preprocessed_path = os.path.join(self.preprocessed_dir, f'{idx}.pkl')

        # Check if this item has been preprocessed
        if self.preprocessed_flags[idx]:
            # If preprocessed, try to load the data from the file
            try:
                with open(preprocessed_path, 'rb') as f:
                    data, label = pickle.load(f)
            except (IOError, EOFError) as e:
                # Handle file reading errors
                raise Exception(f"Error reading file {preprocessed_path}: {e}")
        else:
            # If not preprocessed, preprocess and save the data
            data, label = self._preprocess_and_save(idx)

        return data, label

    def __len__(self):
        """
        Return the length of the dataset.
        Returns:
            Length of the dataset.
        """
        return len(self.dataset)
