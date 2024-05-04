 # -*- coding: utf-8 -*-

"""
Sparsey Input Reshape: file holding the 
    SparseyInputReshapeTransform class.
"""


import torch

from .abstract_transform import AbstractTransform


class SparseyInputReshapeTransform(AbstractTransform):
    """
    A transform to first convert an image to grayscale and then
    binarize it based on a threshold.
    """
    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Reshape the input for Sparsey models.

        Returns:
            (torch.Tensor): the reshaped input.
        """
        return sample.view(sample.shape[0], -1, 1)
