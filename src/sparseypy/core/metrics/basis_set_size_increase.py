import numpy
import torch

from typing import Optional, Callable

from sparseypy.access_objects.models.model import Model
from sparseypy.core.model_layers.sparsey_layer import SparseyLayer
from sparseypy.core.metrics.metrics import Metric
from sparseypy.core.metrics.comparisons import min_by_layerwise_mean


class BasisSetSizeIncreaseMetric(Metric):
    """
    BasisSetSizeIncreaseMetric: metric to keep track
        of basis set sizes across a Sparsey model.
    """
    def __init__(self, model: torch.nn.Module,
                 device: torch.device,
                 reduction: Optional[str] = None,
                 best_value: Optional[Callable] = min_by_layerwise_mean):
        super().__init__(model, "basis_set_size_increase", best_value, device)
        self.old_sizes = self._get_set_sizes(model)

        self.reduction = reduction

    def compute(self, m: Model, last_batch: torch.Tensor,
                labels: torch.Tensor, training: bool = True):

        # non-recursive version will need to be replaced
        # for nested models
        # once we solve the nondeterministic return from m.modules()
        # compute sizes in the same way as BasisSetSize
        new_sizes = self._get_set_sizes(m)

        # if you don't train your own model,
        # this will produce an error on run 0
        # then get the change in basis for each MAC
        #if old_sizes:
        deltas = []

        for layer_index in range(len(new_sizes)):
            deltas.append(
                numpy.subtract(
                    new_sizes[layer_index],
                    self.old_sizes[layer_index]
                )
            )

        # the current basis set is the new previous basis set
        self.old_sizes = new_sizes

        # return the differences
        if self.reduction is None or self.reduction == "none":
            return deltas
        elif self.reduction == 'mean':
            return [
                sum(layer_deltas) / len(layer_deltas) if len(layer_deltas) > 0 else None for layer_deltas in deltas
            ]
        elif self.reduction == 'sum':
            return [
                sum(layer_deltas) for layer_deltas in deltas
            ]
        elif self.reduction == 'highest_layer':
            return deltas[-1]
        else:
            return None
    

    def _get_set_sizes(self, m: Model):
        return [
            [len(mac.stored_codes) for mac in layer.mac_list]
            for layer in m.children() if isinstance(layer, SparseyLayer)
        ]
    