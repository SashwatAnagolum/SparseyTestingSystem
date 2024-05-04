# -*- coding: utf-8 -*-
"""
Preprocessed Dataset: wrapper for datasets
"""
import os
import pickle
from sparseypy.access_objects.datasets.dataset import Dataset
from sparseypy.access_objects.preprocessing_stack.preprocessing_stack import PreprocessingStack

class PreprocessedDataset(Dataset):
    """
    A dataset wrapper class that applies preprocessing to another dataset and caches the results.
    Attributes:
        dataset (Dataset): The original dataset to be preprocessed.
        preprocessed_dir (str): Directory where preprocessed data is stored.
        preprocessing_stack (PreprocessingStack): The preprocessing operations to be applied.
        preprocessed_flags (list[bool]): A boolean list indicating whether an item has been preprocessed.
    """

    def __init__(self, dataset: Dataset,
                 preprocessing_stack: PreprocessingStack,
                 preprocessed_dir: str = "datasets/preprocessed_dataset",
                 save_to_disk: bool = False):
        """
        Initialize the PreprocessedDataset.

        Args:
            dataset (Dataset): The dataset to be preprocessed.
            preprocessed_dir (str): Directory to store preprocessed data.
            preprocessing_stack (PreprocessingStack): Stack of preprocessing steps to apply.
            save_to_disk (bool): 
        """
        super().__init__()

        self.dataset = dataset
        self.preprocessed_dir = preprocessed_dir
        self.preprocessing_stack = preprocessing_stack
        self.save_to_disk = save_to_disk

        # Create the directory for preprocessed data if it does not exist
        if self.save_to_disk:
            self.saved_to_disk_flags = [False for i in range(len(self.dataset))]
            
            if not os.path.exists(self.preprocessed_dir):
                os.makedirs(self.preprocessed_dir)


    def save_data_to_disk(self, data, label, idx) -> None:
        """
        Save data for a given index.

        Args:
            data (torch.Tensor): the data to save.
            label (torch.Tensor): the label to save.
            idx (int): Index of the data in the dataset.
        """
        # Path where the preprocessed data will be saved
        preprocessed_path = os.path.join(
            self.preprocessed_dir, f'{idx}.pkl'
        )

        # Save the preprocessed data and label
        with open(preprocessed_path, 'wb') as f:
            pickle.dump((data, label), f)

        # Mark this item as preprocessed in the boolean array
        self.saved_to_disk_flags[idx] = True


    def __getitem__(self, idx):
        """
        Get item by index, applying preprocessing if necessary.
        Args:
            idx (int): Index of the data.
        Returns:
            Preprocessed data and its label.
        """
        if self.save_to_disk:
            # Path to the preprocessed file for this index
            preprocessed_path = os.path.join(
                self.preprocessed_dir, f'{idx}.pkl'
            )

            if self.saved_to_disk_flags[idx]:
                # If preprocessed, try to load the data from the file
                try:
                    with open(preprocessed_path, 'rb') as f:
                        data, label = pickle.load(f)
                except (IOError, EOFError) as e:
                    # Handle file reading errors
                    raise Exception(
                        f"Error reading file {preprocessed_path}: {e}"
                    ) from e
            else:
                # If not preprocessed, preprocess and save the data
                data, label = self.dataset.__getitem__(idx)
                data = self.preprocessing_stack(data)

                self.save_data_to_disk(data, label, idx)
        else:
            data, label = self.dataset.__getitem__(idx)
            data = self.preprocessing_stack(data)

        return data, label


    def __len__(self):
        """
        Return the length of the dataset.
        Returns:
            Length of the dataset.
        """
        return len(self.dataset)
