# -*- coding: utf-8 -*-

"""
Num Activations: file holding the NumActivationsMetric class.
"""


from typing import Optional, Callable

import torch

from sparseypy.access_objects.models.model import Model
from sparseypy.core.hooks import LayerIOHook
from sparseypy.core.metrics.metrics import Metric
from sparseypy.core.metrics.comparisons import min_by_layerwise_mean


class NumActivationsMetric(Metric):
    """
    NumActivationsMetric: metric computing the number of activations
        across MACs in a Sparsey model.

    Attributes:
        reduction (str): the type of reduction to apply
            onto the raw per-layer, per-sample feature coverage
            results.
        hook (LayerIOHook): the hook registered with the model
            being evaluated to obtain references to each layer,
            and layerwise inputs and outputs.
    """
    def __init__(self, model: torch.nn.Module,
                 device: torch.device,
                 reduction: Optional[str] = None,
                 best_value: Optional[Callable] = min_by_layerwise_mean) -> None:
        """
        Initializes the NumActivationsMetric object. 

        Args:
            model (torch.nn.Module): the model to compute feature
                coverage using.
            reduction (Optional[str]): the type of reduction
                to apply before returning the metric value. 
                Valid options are 'layerwise_mean', 'sum',
                'mean', 'none', and None.
        """
        super().__init__(
            model, 'num_activations', best_value,
            device, reduction
        )

        self.reduction = reduction
        self.hook = LayerIOHook(self.model)


    def _compute(self, m: Model, last_batch: torch.Tensor,
                labels: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Computes the number of activations of a model for a given batch of inputs.

        Args:
            m (Model): Model to evaluate.
            last_batch (torch.Tensor): the model input for the current step
            labels (torch.Tensor): the labels for the current step
            training (bool): whether the model is training or evaluating

        Output:
            Union[float | list[float] | list[list[float]]]:
                the number of activations across MACs in the model
        """
        layers, _, _ = self.hook.get_layer_io()

        num_activations = []

        for layer in layers:
            num_activations.append(
                layer.is_active.clone()
            )

        return torch.nested.nested_tensor(
            num_activations, dtype=torch.float32,
            device=self.device
        )
