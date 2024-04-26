"""
comparsions.py - contains comparison functions for determining the "best" value of a metric
"""

import numpy as np
import torch

#### SUPPORT FUNCTIONS ####
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
    if isinstance(data, torch.Tensor):
        if data.ndim == 2:  
            return torch.mean(
                torch.stack(
                    [
                        torch.mean(list)
                        for list in data.unbind()
                    ]
                )
            ).cpu()
        elif data.ndim == 1:
            return torch.mean(data).cpu().item()
        else:
            return data.cpu().item()
    elif isinstance(data, list):
        if len(data) == 0:
            data=[0]
        ret = np.mean(
            np.nan_to_num(
                [average_nested_data(item) for item in data]
            )
        )
    elif hasattr(data, 'tolist'):  # numpy array
        if len(data) == 0:
            data=[0]
        ret = np.mean(np.nan_to_num(data))
    else:
        # Scalar value
        ret = data

    return ret.item() if isinstance(ret, np.generic) else ret


#### BUILT IN COMPARISON FUNCTIONS ####
def max_by_layerwise_mean(x, y):
    """
    Returns the maximum value by layerwise average of x and y.
    """
    return average_nested_data(x) > average_nested_data(y)


def min_by_layerwise_mean(x, y):
    """
    Returns the minimum value by layerwise average of x and y.
    """
    return average_nested_data(x) < average_nested_data(y)


#### CUSTOM COMPARISON FUNCTIONS ####
