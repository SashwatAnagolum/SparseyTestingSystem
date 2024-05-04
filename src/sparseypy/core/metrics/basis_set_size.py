import torch

from typing import Optional, Callable

from sparseypy.core.metrics.metrics import Metric
from sparseypy.core.hooks import LayerIOHook
from sparseypy.core.metrics.comparisons import min_by_layerwise_mean
from sparseypy.access_objects.models.model import Model


class BasisSetSizeMetric(Metric):
    """
    Basis Set Size Metric: computes the size of the basis set
        for each MAC in the model.
    """
    def __init__(self, model: torch.nn.Module,
                 device: torch.device,
                 reduction: Optional[str] = None,
                 best_value: Optional[Callable] = min_by_layerwise_mean):
        """
        Initializes the Basis Set Size Metric.
        Args:
            model (torch.nn.Module): the model to compute the metric on.
            device (torch.device): the device to compute the metric on.
            reduction (str): the reduction method to use.
            best_value (function): the best value function to use.
        """
        super().__init__(
            model, 'basis_set_size', best_value,
            device, reduction
        )

        self.projections = None
        self.codes = None
        self.hook = LayerIOHook(self.model)


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
        last_batch = last_batch.view(last_batch.shape[0], -1)

        layers, _, outputs = self.hook.get_layer_io()
        batch_size = last_batch.shape[0]
        last_batch = last_batch.view(batch_size, -1)

        if self.projections is None:
            self.projections = [
                torch.randint(
                    int(-1 * 1e6), int(1e6),
                    (output.shape[2], 1), dtype=torch.float32,
                    device=self.device
                ) for output in outputs
            ]

            self.codes = [
                [set() for i in range(output.shape[1])]
                for output in outputs
            ]

        if training:
            for layer_index, layer in enumerate(layers):
                proj_codes = torch.matmul(
                    outputs[layer_index],
                    self.projections[layer_index]
                ).int().view(batch_size, -1)

                for image_index in range(batch_size):
                    for mac_index in range(layer.is_active.shape[1]):
                        if layer.is_active[image_index][mac_index]:
                            self.codes[layer_index][mac_index].add(
                                proj_codes[image_index][mac_index].item()
                            )

            basis_set_sizes = [
                [len(s) for s in self.codes[layer_index]]
                for layer_index in range(len(layers))
            ]
        else:
            basis_set_sizes = [
                [0 for s in self.codes[layer_index]]
                for layer_index in range(len(layers))
            ]

        basis_set_sizes = [
            torch.tensor(
                b, dtype=torch.float32,
                device=self.device
            ).unsqueeze(0).expand(batch_size, len(b))
            for b in basis_set_sizes
        ]

        return torch.nested.nested_tensor(
            basis_set_sizes, dtype=torch.float32, device=self.device
        )