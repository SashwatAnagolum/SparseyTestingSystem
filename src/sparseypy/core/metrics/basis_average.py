# -*- coding: utf-8 -*-

"""
Basis Average: file holding the BasisAverageMetric class.
"""


from typing import Optional, Callable

import torch

from sparseypy.access_objects.models.model import Model
from sparseypy.core.hooks import LayerIOHook
from sparseypy.core.metrics.metrics import Metric
from sparseypy.core.metrics.comparisons import max_by_layerwise_mean


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
                 device: torch.device,
                 reduction: Optional[str] = None,
                 best_value: Optional[Callable] = max_by_layerwise_mean) -> None:
        """
        Initializes the BasisAverageMetric object. 

        Args:
            model (torch.nn.Module): the model to compute feature
                coverage using.
            reduction (Optional[str]): the type of reduction
                to apply before returning the metric value.
        """
        super().__init__(
            model, "basis_average", best_value,
            device, reduction
        )

        self.hook = LayerIOHook(self.model)
        self.summed_inputs = None
        self.num_inputs_seen = None
        self.projected_rfs = None
        self.expected_input_shape = None
        self.layer_sizes = []


    def get_projected_receptive_fields(self,
        layers: list[torch.nn.Module],
        outputs: list[torch.Tensor],
        input_shape: int) -> list[torch.Tensor]:
        """
        Compute the projected receptive fields of each MAC in the model,
        i.e. what input elements in each sample can be seen by each MAC.

        Args:
            layers (list[list[MAC]]): collection of MACS in the model.
            input_shape (int): shape of each input sample.
        """
        projected_rfs = [
            torch.zeros(
                (1, output.shape[1] + 1, input_shape + 1),
                dtype=torch.float32,
                device=self.device
            ) for output in outputs
        ]

        projected_rfs[0].scatter_(
            2, layers[0].input_connections.view(
                1, *layers[0].input_connections.shape
            ), torch.ones(
                (1, 1, 1),
                dtype=torch.float32,
                device=self.device
            ).expand(
                1,
                outputs[0].shape[1],
                layers[0].input_connections.shape[-1]
            )
        )

        for i in range(1, len(layers)):
            projected_rfs[i][:, :-1] = torch.sum(
                projected_rfs[i - 1][:, layers[i].input_connections],
                dim=2
            )

        projected_rfs = [
            p[:, :-1, :-1].ge(
                torch.ones(
                    (1), dtype=torch.float32,
                    device=self.device
                )
            )
            for p in projected_rfs
        ]

        return projected_rfs


    def initialize_shapes(self, layers: list[torch.nn.Module],
                          outputs: list[torch.Tensor],
                          last_batch: torch.Tensor) -> None:
        """
        Initialize the shapes of different storage objects in the model
        based on the shape of the inputs and the model structure.

        Args:
            layers (list[list[MAC]]): collection of MACs making up
                the model.
            last_batch (torch.Tensor): the last set of inputs shown
                to the model.
        """
        self.expected_input_shape = int(
            last_batch.numel() / last_batch.shape[0]
        )

        self.projected_rfs = self.get_projected_receptive_fields(
            layers, outputs, self.expected_input_shape
        )

        self.summed_inputs = [
            torch.zeros(
                (1, output.shape[1], self.expected_input_shape), 
                dtype=torch.float32,
                device=self.device
            ) for output in outputs
        ]

        self.num_inputs_seen = [
            torch.zeros(
                (1, output.shape[1]),
                dtype=torch.float32,
                device=self.device
            ) for output in outputs
        ]


    def _compute(self, m: Model, last_batch: torch.Tensor,
                 labels: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Computes a metric.

        Args:
            m (torch.nn.Module): the model currently being trained.
            last_batch (torch.Tensor): the inputs to the
                current batch being evaluated
            labels (torch.Tensor): the output from the
                current batch being evaluated

        Returns:
            (torch.Tensor): the raw metric values.     
        """
        layers, _, outputs = self.hook.get_layer_io()
        last_batch = last_batch.view(last_batch.shape[0], 1, -1)
        batch_size = last_batch.shape[0]

        if self.num_inputs_seen is None:
            self.initialize_shapes(layers, outputs, last_batch)

        if training:
            for layer_index, layer in enumerate(layers):
                proj_inputs = torch.mul(
                    last_batch, self.projected_rfs[layer_index]
                )

                torch.mul(
                    proj_inputs,
                    layer.is_active.view(*layer.is_active.shape, 1),
                    out=proj_inputs
                )

                torch.add(
                    self.summed_inputs[layer_index],
                    proj_inputs.sum(0, keepdim=True),
                    out=self.summed_inputs[layer_index]
                )

                torch.add(
                    self.num_inputs_seen[layer_index],
                    layer.is_active.sum(0, keepdim=True),
                    out=self.num_inputs_seen[layer_index]
                )

            basis_averages = [
                torch.nan_to_num(
                    torch.div(basis, num_inputs.unsqueeze(-1)),
                    0.0
                ) for basis, num_inputs in zip(
                    self.summed_inputs,
                    self.num_inputs_seen
                )
            ]

            basis_averages = [
                b.expand(batch_size, *b.shape[1:])
                for b in basis_averages
            ]
        else:
            basis_averages = [
                torch.zeros(
                    (1, output.shape[1], last_batch.shape[1]),
                    dtype=torch.float32,
                    device=self.device
                ).expand(batch_size, output.shape[1], last_batch.shape[1])
                for output in outputs
            ]

        return torch.nested.nested_tensor(
            basis_averages, dtype=torch.float32, device=self.device
        )
