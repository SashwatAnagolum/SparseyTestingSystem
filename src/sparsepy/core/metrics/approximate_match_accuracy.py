import abc

import torch

from typing import Optional

from sparsepy.access_objects.models.model import Model
from sparsepy.core.model_layers.sparsey_layer import MAC
from sparsepy.core.hooks import LayerIOHook
from sparsepy.core.metrics.metrics import Metric

class ApproximateMatchAccuracyMetric(Metric):

    def __init__(self, model: torch.nn.Module, reduction: Optional[str] = None):
        super().__init__(model)
        # attaches the hook anew for this Metric to gain access to the hook data
        # consider hook managerlater if we need to use many metrics with hooks
        self.hook = LayerIOHook(self.model)
        # initialize input map
        self.stored_inputs = {}

        self.reduction = reduction


    def compute(self, m: Model, last_batch: torch.Tensor, labels: torch.Tensor, training: bool = True):
        """
        Computes the approximate match accuracy of a model for a given batch of inputs.

        Args:
            m: Model to evaluate.
            last_batch: the model input for the current step (as a Tensor)
            labels: the model output for the current step (as a Tensor)
            training: boolean - whether the model is training (store codes) or evaluating (determine approximate match accuracy using stored codes)

        Output:
            approximate match accuracy as a fraction.
        """

        # fetch the outputs from each model layer using the hook so we can get the codes for the current input item
        layers, layer_inputs, layer_outputs = self.hook.get_layer_io()
        
        # start the match counts
        hits = 0
        total = 0

        # loop over batch items
        for image in last_batch:
            # construct dict keys by flattening image tensor to 1D, converting to ints, and concatenating to a string
            image_str = "".join([str(i.item()) for i in image.flatten().int()])

            # if we have not seen this input before, and we are training, we should store it
            if image_str not in self.stored_inputs.keys():
                if training:
                    self.stored_inputs[image_str] = layer_outputs
            # if we have seen this input
            else:
                # and we are also evaluating
                if not training:
                    # then we need to determine whether the codes are the same
                    # for every layer in the output
                    for layer_index in range(len(layer_outputs)):
                        # and for every MAC in that layer
                        for mac_index in range(len(layer_outputs[layer_index])):


                            # increment number of MACs evaluated
                            #total += 1
                            # determine whether the codes are the same
                            #approximate_match = torch.allclose(self.stored_inputs[image_str][layer_index][mac_index], layer_outputs[layer_index][mac_index], atol=1e-5)
                            # if so, increment hits
                            #if approximate_match:
                            #    hits += 1

                            ####CHECK CHANGES HERE

                            #get number of neurons in current MAC
                            num_neurons = layer_outputs[layer_index][mac_index].shape[-1]

                            #itrate over each neuron
                            for neuron_index in range(num_neurons):
                                #increment total neurons evaluated
                                total += 1

                                #determine whether the codes are the same
                                approximate_match = torch.allclose(self.stored_inputs[image_str][layer_index][mac_index][...,neuron_index],layer_outputs[layer_index][mac_index][...,neuron_index], atol=1e-5)


                                #If so, increment hits
                                if approximate_match:
                                    hits += 1
        
        # BUG reduction is already mean across all MACs by default
        # return final approximate match accuracy as a fraction
        return 0 if total == 0 else hits / total