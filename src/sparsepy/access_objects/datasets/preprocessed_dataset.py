
"""
Preprocess Dataset: file holding the image dataset class.
"""
import os
import torch
import pickle
from sparsepy.access_objects.datasets.dataset import Dataset
from sparsepy.access_objects.preprocessing_stack.preprocessing_stack import PreprocessingStack

class PreprocessedDataset(Dataset):
    """
    A dataset wrapper class that applies preprocessing to another dataset and caches the results.

    Attributes:
        dataset (Dataset): The original dataset to be preprocessed.
        preprocessed_dir (str): Directory where preprocessed data is stored.
        preprocessing_stack (PreprocessingStack): The preprocessing operations to be applied.
    """
    def __init__(self, dataset: Dataset, preprocessed_dir: str, preprocessing_stack: PreprocessingStack):
        self.dataset = dataset
        self.preprocessed_dir = preprocessed_dir
        self.preprocessing_stack = preprocessing_stack

        if not os.path.exists(self.preprocessed_dir):
            os.makedirs(self.preprocessed_dir)

    def _preprocess_and_save(self, idx):
        """
        Preprocesses the data at a given index and saves it.

        Args:
            idx (int): The index of the data in the dataset.

        Returns:
            The preprocessed data.
        """
        data, label = self.dataset[idx]
        preprocessed_data = self.preprocessing_stack(data)

        preprocessed_path = os.path.join(self.preprocessed_dir, f'{idx}.pkl')
        with open(preprocessed_path, 'wb') as f:
            pickle.dump(preprocessed_data, label, f)

        return preprocessed_data

    def __getitem__(self, idx):
        preprocessed_path = os.path.join(self.preprocessed_dir, f'{idx}.pkl')
        # If preprocessed data exists, load and return it
        if os.path.exists(preprocessed_path):
            with open(preprocessed_path, 'rb') as f:
                data, label = pickle.load(f)
        # If not, preprocess and save the data
        else:
            data, label = self._preprocess_and_save(idx)

        return data

    def __len__(self):
        return len(self.dataset)

