import torch
import numpy

from sparsepy.access_objects.models.model import Model
from sparsepy.core.model_layers.sparsey_layer import MAC, SparseyLayer
from sparsepy.core.hooks import LayerIOHook
from sparsepy.core.metrics.metrics import Metric


class BasisSetSizeIncreaseMetric(Metric):
    """
    BasisSetSizeIncreaseMetric: metric to keep track
        of basis set sizes across a Sparsey model.
    """
    def __init__(self, model: torch.nn.Module):
        super().__init__(model)
        self.old_sizes = self._get_set_sizes(model)

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
        return deltas
    

    def _get_set_sizes(self, m: Model):
        return [
            [len(mac.stored_codes) for mac in layer.mac_list]
            for layer in m.children() if isinstance(layer, SparseyLayer)
        ]
    