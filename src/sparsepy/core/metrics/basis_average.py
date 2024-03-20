# -*- coding: utf-8 -*-

"""
Basis Average: file holding the BasisAverageMetric class.
"""


from typing import Optional

import torch

from sparsepy.access_objects.models.model import Model
from sparsepy.core.hooks import LayerIOHook
from sparsepy.core.metrics.metrics import Metric
from sparsepy.core.metrics.comparisons import max_by_layerwise_mean


class BasisAverageMetric(Metric):
    """
    BasisAverageMetric: metric computing the feature
        coverage of MACs and layers in a Sparsey model.

    Attributes:
        reduction (str): the type of reduction to apply
            onto the raw per-layer, per-sample feature coverage
            results. Valid options are None and 'sparse'. Choosing
            'sparse' will return the raw averaged inputs to each MAC.
            Choosing None will return the inputs inserted into
            their positions in a tensor of the same size as the 
            input samples to the model.
        hook (LayerIOHook): the hook registered with the model
            being evaluated to obtain references to each layer,
            and layerwise inputs and outputs.
    """
    def __init__(self, model: torch.nn.Module,
                 reduction: Optional[str] = None) -> None:
        """
        Initializes the BasisAverageMetric object. 

        Args:
            model (torch.nn.Module): the model to compute feature
                coverage using.
            reduction (Optional[str]): the type of reduction
                to apply before returning the metric value.
        """
        super().__init__(model, "basis_average", max_by_layerwise_mean)

        self.reduction = reduction
        self.hook = LayerIOHook(self.model)
        self.summed_inputs = None
        self.num_inputs_seen = None
        self.projected_rfs = None
        self.expected_input_shape = None


    def get_projected_receptive_fields(self, layers, input_shape) -> None:
        """
        Compute the projected receptive fields of each MAC in the model,
        i.e. what input elements in each sample can be seen by each MAC.

        Args:
            layers (list[list[MAC]]): collection of MACS in the model.
            input_shape (int): shape of each input sample.
        """
        projected_rfs = [
            [
                torch.zeros((input_shape), dtype=torch.bool)
                for j in range(len(layers[i]))
            ] for i in range(len(layers))
        ]

        for mac_index, mac in enumerate(layers[0]):
            projected_rfs[0][mac_index][mac.input_filter] = True

        for layer_num in range(1, len(layers)):
            for mac_index, mac in enumerate(layers[layer_num]):
                for input_source in mac.input_filter:
                    projected_rfs[layer_num][mac_index] = torch.logical_or(
                        projected_rfs[layer_num][mac_index],
                        projected_rfs[layer_num - 1][input_source]
                    )

        return projected_rfs


    def initialize_shapes(self, layers, last_batch) -> None:
        """
        Initialize the shapes of different storage objects in the model
        based on the shape of the inputs and the model structure.

        Args:
            layers (list[list[MAC]]): collection of MACs making up
                the model.
            last_batch (torch.Tensor): the last set of inputs shown
                to the model.
        """
        self.expected_input_shape = last_batch.shape[1]

        self.projected_rfs = self.get_projected_receptive_fields(
            layers, self.expected_input_shape
        )

        self.summed_inputs = [
            [
                torch.zeros(
                    torch.sum(self.projected_rfs[i][j]),
                    dtype=torch.float32
                ) for j in range(len(layers[i]))
            ] for i in range(len(layers))
        ]

        self.num_inputs_seen = [
            [0 for j in range(len(layers[i]))]
            for i in range(len(layers))
        ]


    def compute(self, m: Model, last_batch: torch.Tensor,
                labels: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Computes the feature coverage of a model for a given batch of inputs.

        Args:
            m (Model): Model to evaluate.
            last_batch (torch.Tensor): the model input for the current step
            labels (torch.Tensor): the model output for the current step
            training (bool): whether the model is training or evaluating

        Returns:
            (list[torch.Tensor]): a list of Tensors containing the average
                feature that each MAC has seen.
        """
        layers, _, _ = self.hook.get_layer_io()

        if self.num_inputs_seen is None:
            self.initialize_shapes(layers, last_batch)

        if training:
            last_batch = last_batch.squeeze()

            for layer_index, layer in enumerate(layers):
                for mac_index, mac in enumerate(layer):
                    self.summed_inputs[layer_index][mac_index] += (
                        torch.sum(
                            last_batch[
                                :,
                                self.projected_rfs[layer_index][mac_index]
                            ] * mac.is_active.unsqueeze(1), 0
                        )
                    )

                    self.num_inputs_seen[layer_index][mac_index] += (
                        torch.sum(mac.is_active).item()
                    )

        if self.reduction is None:
            return [
                [
                    torch.zeros(
                        self.expected_input_shape,
                        dtype=torch.float32
                    ).scatter_(
                        0, torch.argwhere(self.projected_rfs[i][j]).squeeze(),
                        torch.nan_to_num(
                            self.summed_inputs[i][j] /
                            self.num_inputs_seen[i][j]
                        )
                    ) for j in range(len(layers[i]))
                ] for i in range(len(layers))
            ]
        else:
            return [
                [
                    torch.nan_to_num(
                        self.summed_inputs[i][j] /
                        self.num_inputs_seen[i][j]
                    ) for j in range(len(layers[i]))
                ] for i in range(len(layers))
            ]
