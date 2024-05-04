import abc
from typing import Callable

import torch
from sparseypy.access_objects.models.model import Model


class Metric(abc.ABC):
    """
    Metric: a base class for metrics.
        Metrics are used to compute different measurements requested by the user
        to provide estimations of model progress and information
        required for Dr. Rinkus' experiments.
    """
    def __init__(self, model: torch.nn.Module, name: str,
                 best_comparison: Callable, device: torch.device,
                 reduction: str) -> None:
        """
        Initializes the Metric object.

        Args:
            model (torch.nn.Module): the model to compute
                the metric for.
            name (str): the name of the metric.
            best_comparison (Callable): the function to use
                for comparing metric value to determine which
                is better.
            device (torch.device): the device to perform
                computation on.
            reduction (str): the type of reduction to 
                apply to the raw metric values computed.
        """
        self.model = model
        self.name = name
        self.best_comparison = best_comparison
        self.device = device
        self.reduction = reduction


    @abc.abstractmethod
    def _compute(self, m: Model, last_batch: torch.Tensor,
                 labels: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Computes a metric.

        Args:
            m: the model currently being trained.

            last_batch: the inputs to the current batch being evaluated

            labels: the output from the current batch being evaluated

        Returns:
            (torch.Tensor): the raw metric values.
        """

    def compute(self, m: Model, last_batch: torch.Tensor,
                 labels: torch.Tensor,
                 training: bool = True) -> torch.Tensor:
        """
        Computes a metric.

        Args:
            m (torch.nn.Module): the model currently being trained.

            last_batch (torch.Tensor): the inputs to the current batch being evaluated

            labels: the output from the current batch being evaluated

        Returns:
            (torch.Tensor): the computed metric.
        """
        raw_values = self._compute(m, last_batch, labels, training)

        return self.reduce_metric_values(raw_values)


    def get_name(self):
        """
        Returns the name of this metric.
        """
        return self.name


    def get_best_comparison_function(self) -> Callable:
        """
        Returns the function to use to obtain the
        "best" instance of this metric.
        """
        return self.best_comparison


    def reduce_metric_values(self, raw_values: torch.Tensor) -> torch.Tensor:
        """
        Performs a reduction (could be the identity)
        on the raw metric values computed, and returns the
        reduced values.

        Args:
            raw_values (torch.Tensor): an optionally
                nested tensor that contains MAC-wise
                values of the metric being computed.
        """
        match self.reduction:
            case 'layerwise_mean':
                metric_val = torch.stack(
                    [
                        torch.mean(t, dim=1, keepdim=True)
                        for t in torch.unbind(raw_values)
                    ]
                )
            case 'layerwise_sum':
                metric_val = torch.stack(
                    [
                        torch.sum(t, dim=1, keepdim=True)
                        for t in torch.unbind(raw_values)
                    ]
                )
            case 'mean':
                metric_val = torch.mean(
                    torch.stack(
                        [
                            torch.mean(t)
                            for t
                            in torch.unbind(raw_values)
                        ]
                    )
                )
            case 'sum':
                metric_val = torch.sum(
                    torch.stack(torch.unbind(raw_values))
                )
            case 'highest_layer':
                metric_val = raw_values[-1]
            case 'highest_layer_mean':
                metric_val = torch.mean(raw_values[-1])
            case _:
                metric_val = raw_values

        return metric_val
        