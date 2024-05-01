"""
comparsions.py - contains comparison functions for determining the "best" value of a metric
"""


from sparseypy.core.metrics.reductions import average_nested_data


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
