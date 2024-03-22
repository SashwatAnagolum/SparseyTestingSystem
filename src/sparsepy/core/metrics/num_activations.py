# -*- coding: utf-8 -*-

"""
Num Activations: file holding the NumActivationsMetric class.
"""


from typing import Optional, Callable

import torch

from sparsepy.access_objects.models.model import Model
from sparsepy.core.hooks import LayerIOHook
from sparsepy.core.metrics.metrics import Metric
from sparsepy.core.metrics.comparisons import max_by_layerwise_mean


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
        num_activations (list[list[int]]): the number of activations
            fr each MAC in each layer of the model.
    """
    def __init__(self, model: torch.nn.Module,
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
        super().__init__(model, "num_activations", best_value)

        self.reduction = reduction
        self.hook = LayerIOHook(self.model)
        self.num_activations = None


    def initialize_activation_counts(self, layers):
        """
        Initializes the activation counts of the NumActivations object.

        Args:
            layers (list[list[MAC]]): a list of MACs in the model.
        """
        self.num_activations = [
            [0 for i in range(len(layer))] for layer in layers
        ]


    def compute(self, m: Model, last_batch: torch.Tensor,
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

        if self.num_activations is None:
            self.initialize_activation_counts(layers)

        for layer_index, layer in enumerate(layers):
            for mac_index, mac in enumerate(layer):
                self.num_activations[
                    layer_index
                ][mac_index] = torch.sum(mac.is_active).item()

        if self.reduction is None or self.reduction == 'none':
            return [
                [mac_activations for mac_activations in layer_activations]
                for layer_activations in self.num_activations
            ]
        elif self.reduction == 'layerwise_mean':
            return [
                sum(layer_activations) / len(layer_activations)
                for layer_activations in self.num_activations    
            ]
        elif self.reduction == 'sum':
            return sum(
                [
                    sum(layer_activations) for layer_activations
                    in self.num_activations
                ]
            )
        else:
            return sum(
                [
                    sum(layer_activations) for layer_activations
                    in self.num_activations
                ]
            ) / sum(
                [
                    len(layer_activations)
                    for layer_activations
                    in self.num_activations
                ]
            )
