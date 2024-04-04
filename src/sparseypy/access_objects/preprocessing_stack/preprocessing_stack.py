# -*- coding: utf-8 -*-

"""
Preprocessing Stack: class to contains ordered lists of PyTorch transforms.
"""


import torch

from sparseypy.core.transforms.transform_factory import TransformFactory


class PreprocessingStack(torch.nn.Module):
    def __init__(self, transform_configs):
        super().__init__()

        self.transform_list = []

        for transform_config in transform_configs['transform_list']:
            transform = TransformFactory.create_transform(
                transform_config['name'],
                **(transform_config['params'] if 'params' in transform_config else {})
            )

            self.transform_list.append(transform)


    def forward(self, x: torch.Tensor):
        for transform in self.transform_list:
            x = transform(x)

        return x
