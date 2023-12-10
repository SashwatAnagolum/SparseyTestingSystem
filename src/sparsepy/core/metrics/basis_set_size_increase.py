import torch
import numpy
from sparsepy.access_objects.models.model import Model
from sparsepy.core.model_layers.sparsey_layer import MAC, SparseyLayer
from sparsepy.core.hooks import LayerIOHook
from sparsepy.core.metrics.metrics import Metric

class BasisSetSizeIncreaseMetric(Metric):

    old_sizes = None

    def __init__(self, model: torch.nn.Module):
        super().__init__(model)
        self.old_sizes = self._get_set_sizes(model)
        #old_sizes = [
        #    [len(mac.stored_codes) for mac in layer.children() if isinstance(mac, MAC)] for layer in model.children()
        #]

    def compute(self, m: Model, last_batch: torch.Tensor, labels: torch.Tensor, training: bool = True):

        # non-recursive version will need to be replaced for nested models
        # once we solve the nondeterministic return from m.modules()
        # compute sizes in the same way as BasisSetSize
        new_sizes = self._get_set_sizes(m)

        #new_sizes = [
        #    [len(mac.stored_codes) for mac in layer.children() if isinstance(mac, MAC)] for layer in m.children()
        #]

        # if you don't train your own model, this will produce an error on run 0
        # then get the change in basis for each MAC
        #if old_sizes:
        deltas = numpy.subtract(new_sizes, self.old_sizes)
        #else:
        #    deltas = new_sizes

        # the current basis set is the new previous basis set
        self.old_sizes = new_sizes

        # return the differences
        return deltas
    
    def _get_set_sizes(self, m: Model):
        #return [
        #    [(len(gc.stored_codes) or 0) for gc in child.children() if isinstance(gc, MAC)] for child in m.children() if isinstance(child, SparseyLayer)
        #]
        return [[len(mac.stored_codes) for mac in layer.mac_list] for layer in m.children() if isinstance(layer, SparseyLayer)]
    