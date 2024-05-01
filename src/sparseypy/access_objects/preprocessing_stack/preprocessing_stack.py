# -*- coding: utf-8 -*-

"""
Preprocessing Stack: class to contains ordered lists of PyTorch transforms.
"""


import torch

from sparseypy.core.transforms.transform_factory import TransformFactory


class PreprocessingStack(torch.nn.Module):
    """
    A class to represent a stack of transforms
    Attributes:
        transform_list (list[torch.nn.Module]): The list of transforms to apply.
    """
    def __init__(self, transform_configs):
        """
        Initializes the PreprocessingStack.
        Args:
            transform_configs (dict): A dictionary containing the list of transforms to apply.
        """
        super().__init__()

        self.transform_list = []

        for transform_config in transform_configs['transform_list']:
            transform = TransformFactory.create_transform(
                transform_config['name'],
                **(transform_config['params'] if 'params' in transform_config else {})
            )

            self.transform_list.append(transform)


    def forward(self, x: torch.Tensor):
        """
        Forward pass through the preprocessing stack.
        Args:
            x (torch.Tensor): The input data.
        Returns:
            (torch.Tensor): The transformed data.
        """
        for transform in self.transform_list:
            x = transform(x)

        return x
