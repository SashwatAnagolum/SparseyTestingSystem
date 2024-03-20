import inspect

from sparsepy.core import metrics
from sparsepy.core.metrics.metrics import Metric
import sparsepy.core.metrics.comparisons as comparisons


class MetricFactory:
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
    def create_metric(metric_name, **kwargs) -> Metric:
        """
        Creates a layer passed in based on the layer name and kwargs.
        """
        metric_class = MetricFactory.get_metric_class(metric_name)

        if "best_value" in kwargs.keys():
            kwargs["best_value"] = getattr(comparisons, kwargs["best_value"])

        metric_obj = metric_class(**kwargs)

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
