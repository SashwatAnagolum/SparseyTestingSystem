import torch

from typing import Optional, Callable

from sparseypy.core.metrics.metrics import Metric
from sparseypy.core.metrics.comparisons import min_by_layerwise_mean
from sparseypy.core.model_layers.sparsey_layer import SparseyLayer
from sparseypy.access_objects.models.model import Model


class BasisSetSizeMetric(Metric):
    def __init__(self, model: torch.nn.Module,
                 reduction: Optional[str] = None,
                 best_value: Optional[Callable] = min_by_layerwise_mean):
        super().__init__(model, "basis_set_size", best_value)

        self.reduction = reduction


    def compute(self, m: Model, last_batch: torch.Tensor,
                labels: torch.Tensor, training: bool = True):
        basis_set_sizes = [
            [len(mac.stored_codes) for mac in layer.mac_list]
            for layer in m.children() if isinstance(layer, SparseyLayer)
        ]

        if self.reduction is None or self.reduction == "none":
            return basis_set_sizes
        elif self.reduction == 'mean':
            return [
                sum(layer_basis) / len(layer_basis)
                if len(layer_basis) > 0 else None
                for layer_basis in basis_set_sizes
            ]
        elif self.reduction == 'sum':
            return [
                sum(layer_basis) for layer_basis in basis_set_sizes
            ]
        elif self.reduction == 'highest_layer':
            return basis_set_sizes[-1]
        else:
            return None
