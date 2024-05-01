# -*- coding: utf-8 -*-

"""
Reductions: file holding reduction functions for metric values.
"""


import torch


def average_nested_data(data: torch.Tensor):
    """
    Averages an arbitrarily deep data structure
    and returns the result as a single value.

    Used here to reduce the granularity of data in order
    to store a single value for each step in W&B.

    Args:
        data (torch.Tensor): the (possibly nested) tensor
            containing the raw metric values computed.

    Returns:
        (float): a single value representing the averaged data
    """
    return torch.mean(
        torch.stack([torch.mean(t) for t in data.unbind()])
    ).cpu().item()
