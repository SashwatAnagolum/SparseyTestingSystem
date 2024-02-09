from sparsepy.core import metrics
from sparsepy.core.metrics.metrics import Metric


class MetricFactory:
    allowed_modules = set([i for i in dir(metrics) if i[:2] != '__'])

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

        metric_obj = metric_class(**kwargs)

        return metric_obj