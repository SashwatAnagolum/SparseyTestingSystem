import torch
from sparsepy.core.metrics.metrics import Metric
from sparsepy.core.model_layers.sparsey_layer import MAC, SparseyLayer
from sparsepy.access_objects.models.model import Model

class BasisSetSizeMetric(Metric):

    def __init__(self, model: torch.nn.Module):
        super().__init__(model)

    def compute(self, m: Model, last_batch: torch.Tensor, labels: torch.Tensor, training: bool = True):

        basis_set_sizes = [[len(mac.stored_codes) for mac in layer.mac_list] for layer in m.children() if isinstance(layer, SparseyLayer)]

        return basis_set_sizes