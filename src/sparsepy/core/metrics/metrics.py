import abc

import torch
import numpy
from sparsepy.access_objects.models.model import Model
from sparsepy.core.model_layers import MAC
from sparsepy.core.hooks import LayerIOHook

class Metric:
    """
    Metric: a base class for metrics.
        Metrics are used to compute different measurements requested by the user
        to provide estimations of model progress and information
        required for Dr. Rinkus' experiments.
    """

    model = None
    hook = None

    def __init__(self, model: torch.nn.Module):
        self.model = model

    @abc.abstractmethod
    def compute(self, m: Model, last_batch: torch.Tensor, labels: torch.Tensor, training: bool = True):
        """
        Computes a metric.

        Args:
            m: the model currently being trained.

            last_batch: the inputs to the current batch being evaluated

            labels: the output from the current batch being evaluated

        Returns:
            the Metric's results as a dict.
        """

class ExactMatchAccuracyMetric(Metric):

    horror = {}

    def __init__(self, model: torch.nn.Module):
        super().__init__(model)
        # attaches the hook anew for this Metric to gain access to the hook data
        # this is strongly suboptimal for performance and needs fixing
        self.hook = LayerIOHook(self.model)

    def compute(self, m: Model, last_batch: torch.Tensor, labels: torch.Tensor, training: bool = True):
        layers, layer_inputs, layer_outputs = self.hook.get_layer_io()

        first_layer_input = layer_inputs[0]

        if first_layer_input not in self.horror.keys():
            if training:
                self.horror[first_layer_input] = layer_outputs
            return False
        else:
            return layer_outputs == self.horror[first_layer_input]
        #for layer, layer_input, layer_output in zip(
        #    layers, layer_inputs, layer_outputs
        #):
            # foreach nn.Module layer in the combined list
            #  if that layer is a MAC
            #   sparsify its output
            #   see if that is already in the dictionary
            #   if training:
            # <<does passing the entire input to each MAC mean we need to filter down to the receptive field?
            # or should we just store inputs/list of codes on a layer level
            # >>
            #       add to MAC code data structure if it doesn't already exist
            #   else:
            #       adjust accuracy number accordingly?        

        # foreach module in the hook layer list
        # layer:
        #   mac:
        #     [list of inputs: outputs]

        

class BasisSetSizeMetric(Metric):

    def __init__(self, model: torch.nn.Module):
        super().__init__(model)

    def compute(self, m: Model, last_batch: torch.Tensor, labels: torch.Tensor, training: bool = True):
        
        #basis_set_size = 0
        
        #for layer in m.children():
        #    layer_basis = [x.stored_codes.size for x in layer if isinstance(x ,MAC)]

        # non-recursive version will need to be replaced for nested models
        # once we solve the nondeterministic return from m.modules()
        basis_set_sizes = [
            [mac.stored_codes.size for mac in layer if isinstance(mac, MAC)] for layer in m.children()
        ]

        #for x in m.modules():
        #    if isinstance(x, MAC):
        #        basis_set_size += x.stored_codes.size

        return basis_set_sizes


class BasisSetSizeIncreaseMetric(Metric):

    old_sizes = None

    def __init__(self, model: torch.nn.Module):
        super().__init__(model)
        old_sizes = [
            [mac.stored_codes.size for mac in layer if isinstance(mac, MAC)] for layer in model.children()
        ]

    def compute(self, m: Model, last_batch: torch.Tensor, labels: torch.Tensor, training: bool = True):

        # non-recursive version will need to be replaced for nested models
        # once we solve the nondeterministic return from m.modules()
        # compute sizes in the same way as BasisSetSize
        new_sizes = [
            [mac.stored_codes.size for mac in layer if isinstance(mac, MAC)] for layer in m.children()
        ]

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
    
    def _get_set_sizes(m: Model):
        return [
            [mac.stored_codes.size for mac in layer if isinstance(mac, MAC)] for layer in m.children()
        ]