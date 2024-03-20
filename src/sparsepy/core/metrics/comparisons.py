"""
comparsions.py - contains comparison functions for determining the "best" value of a metric
"""

import numpy as np

#### SUPPORT FUNCTIONS ####
def average_nested_data(data):
    """
    Averages an arbitrarily deep data structure and returns the result as a single value.
    """
    if isinstance(data, list):
        if len(data) == 0:
            data=[0]
        ret = np.mean(np.nan_to_num([average_nested_data(item) for item in data]))
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
    return x if average_nested_data(x) >= average_nested_data(y) else y


def min_by_layerwise_mean(x, y):
    """
    Returns the minimum value by layerwise average of x and y.
    """
    return x if average_nested_data(x) <= average_nested_data(y) else y


#### CUSTOM COMPARISON FUNCTIONS ####
