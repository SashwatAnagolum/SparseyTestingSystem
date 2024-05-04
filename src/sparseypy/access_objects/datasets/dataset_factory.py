# -*- coding: utf-8 -*-

"""
Dataset Factory: file holding the Dataset Factory class.
"""


from torch.utils.data import Dataset

from sparseypy.access_objects import datasets
from sparseypy.access_objects.datasets import PreprocessedDataset, InMemoryDataset
from sparseypy.access_objects.preprocessing_stack.preprocessing_stack import PreprocessingStack


class DatasetFactory:
    """
    Factory class for creating datasets.
    Attributes:
        allowed_modules (set): A set of allowed modules to create datasets from.
    """
    allowed_modules = set([i for i in dir(datasets) if i[:2] != '__'])

    @staticmethod
    def get_dataset_class(dataset_type: str):
        """
        Gets the class corresponding to the name passed in.
        Throws an error if the name is not valid.

        Args:
            dataset_type (str): the type of dataset to create.
        """
        class_name = ''.join(
            [l.capitalize() for l in dataset_type.split('_')] + ['Dataset']
        )

        if class_name in DatasetFactory.allowed_modules:
            return getattr(datasets, class_name)
        else:
            raise ValueError('Invalid dataset type!')


    @staticmethod
    def create_dataset(dataset_type: str, **kwargs) -> Dataset:
        """
        Creates a layer passed in based on the layer name and kwargs.

        Args:
            dataset_type (str) the type of dataset to create.
        """
        dataset_class = DatasetFactory.get_dataset_class(dataset_type)

        dataset_obj = dataset_class(**kwargs)

        return dataset_obj
    

    @staticmethod
    def build_and_wrap_dataset(dataset_config: dict) -> Dataset:
        """
        Builds a dataset and wraps it with appropriate
        wrapper classes specified in the dataset config.
        """
        dataset = DatasetFactory.create_dataset(
            dataset_config['dataset_type'],
            **dataset_config['params']
        )

        if dataset_config['preprocessed'] is True:
            preprocessed_dataset_stack = PreprocessingStack(
                dataset_config['preprocessed_stack']
            )

            dataset = PreprocessedDataset(
                dataset, preprocessed_dataset_stack,
                dataset_config['preprocessed_temp_dir'],
                dataset_config['save_to_disk']
            )

        if dataset_config['in_memory']:
            dataset = InMemoryDataset(
                dataset, dataset_config['load_lazily']
            )

        return dataset
