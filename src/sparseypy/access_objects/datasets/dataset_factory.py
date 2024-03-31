# -*- coding: utf-8 -*-

"""
Dataset Factory: file holding the Dataset Factory class.
"""


from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision import datasets as tv_datasets

from sparseypy.access_objects import datasets


class DatasetFactory:
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
    def get_builtin_dataset_class(dataset_type: str):
        """
        Gets the Torchvision or other builtin class corresponding
        to the name passed in.

        Throws an error if the name is not valid.

        Args:
            dataset_type (str): the type of dataset to create.
        """
        if dataset_type in dir(tv_datasets):
            return getattr(tv_datasets, dataset_type)
        else:
            raise ValueError("Invalid built-in dataset type!")

    @staticmethod
    def create_dataset(dataset_type: str, **kwargs) -> Dataset:
        """
        Creates a layer passed in based on the layer name and kwargs.

        Args:
            dataset_type (str) the type of dataset to create.
        """
        if dataset_type == "built_in":
            dataset_class = DatasetFactory.get_builtin_dataset_class(kwargs.pop("name"))
            kwargs["transform"] = v2.PILToTensor()
        else:
            dataset_class = DatasetFactory.get_dataset_class(dataset_type)

        dataset_obj = dataset_class(**kwargs)

        return dataset_obj

    @staticmethod
    def is_valid_builtin_dataset(name: str) -> bool:
        """
        Checks whether a given dataset name exists as a Torchvision built-in dataset.

        Args:
            name (str) the name of the dataset
        """
        return name in dir(tv_datasets)
