# -*- coding: utf-8 -*-

"""
Dataset Factory: file holding the Dataset Factory class.
"""


import torch

from torch.utils.data import Dataset

from .. import datasets


class DatasetFactory:
    allowed_modules = set([i for i in dir(datasets) if i[:2] != '__'])

    @staticmethod
    def get_dataset_class(dataset_type):
        """
        Gets the class corresponding to the name passed in.
        Throws an error if the name is not valid.
        """
        class_name = ''.join(
            [l.capitalize() for l in dataset_type.split('_')] + ['Dataset']
        )

        if class_name in DatasetFactory.allowed_modules:
            return getattr(datasets, class_name)
        else:
            raise ValueError('Invalid dataset type!')
    

    @staticmethod
    def create_dataset(dataset_type, **kwargs) -> Dataset:
        """
        Creates a layer passed in based on the layer name and kwargs.
        """
        dataset_class = DatasetFactory.get_dataset_class(dataset_type)

        dataset_obj = dataset_class(**kwargs)

        return dataset_obj