"""
metric_factory.py - module containing the MetricFactory
"""
import inspect
from typing import Callable, Optional

import torch

from sparseypy.core import metrics
from sparseypy.core.metrics.metrics import Metric
import sparseypy.core.metrics.comparisons as comparisons


class MetricFactory:
    """
    MetricFactory: Factory class for validating and constructing Metrics and comparison functions.

    Provides methods for validating the existence of and instantiating system Metric classes
    and comparison functions using reflection.
    
    Attributes:
        allowed_comparisons (set[str]): the names of all the valid comparison functions available
            as part of the system.
        allowed_modules (set[str]): the names of all the valid metrics available as part of
            the system.
    """
    allowed_modules = set([i for i in dir(metrics) if i[:2] != '__'])
    allowed_comparisons = set([i[0] for i in inspect.getmembers(comparisons, inspect.isfunction)])

    @staticmethod
    def get_metric_class(metric_name):
        """
        Gets the class corresponding to the name passed in.
        Throws an error if the name is not valid.
        """
        class_name = ''.join(
            [l.capitalize() for l in metric_name.split('_')] + ['Metric']
        )

        if class_name in MetricFactory.allowed_modules:
            return getattr(metrics, class_name)
        # not implemented yet - wrapping PyTorch metrics requires additional consideration
        # (and finding a way in PyTorch to determine what functions count as metrics)
        #elif metric_name in dir(torch.optim):
        #    return getattr(torch.optim, opt_name)
        else:
            raise ValueError('Invalid metric name!')


    @staticmethod
    def create_metric(metric_name: str, reduction: str,
        comparison: Optional[str], params: dict, model: torch.nn.Module,
        device: torch.device) -> Metric:
        """
        Creates a layer passed in based on the metric name and kwargs.
        """
        metric_class = MetricFactory.get_metric_class(metric_name)

        if comparison:
            comparison = getattr(comparisons, comparison)

        metric_obj = metric_class(
            model, device, reduction,
            comparison, **params
        )

        return metric_obj

    @staticmethod
    def is_valid_metric_class(metric_name: str) -> bool:
        """
        Checks whether a metric class exists corresponding to the passed-in name.
        """
        class_name = ''.join(
            [l.capitalize() for l in metric_name.split('_')] + ['Metric']
        )

        return class_name in MetricFactory.allowed_modules

    @staticmethod
    def is_valid_comparision(comparison_name: str) -> bool:
        """
        Checks whether a given comparison function exists.
        """
        return comparison_name in MetricFactory.allowed_comparisons

    @staticmethod
    def get_comparison_function(comparison_name: str) -> Callable:
        """
        Gets the comparison function corresponding to the name passed in.
        Throws an error if the name is not valid.
        args:
            comparison_name (str): The name of the comparison function.
        """
        if comparison_name in MetricFactory.allowed_comparisons:
            return getattr(comparisons, comparison_name)
        else:
            raise ValueError("Invalid comparison function name!")
