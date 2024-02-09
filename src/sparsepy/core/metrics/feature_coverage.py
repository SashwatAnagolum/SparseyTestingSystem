import abc
import numpy as np
import torch

from sparsepy.access_objects.models.model import Model
from sparsepy.core.model_layers.sparsey_layer import MAC
from sparsepy.core.hooks import LayerIOHook
from sparsepy.core.metrics.metrics import Metric

class FeatureCoverageMetric(Metric):

    def __init__(self, model: torch.nn.Module):
        super().__init__(model)
        # attaches the hook anew for this Metric to gain access to the hook data
        # to check "every code at every level" we require access to the inner model data to determine which MACs have been activated
        self.hook = LayerIOHook(self.model)


    def compute(self, m: Model, last_batch: torch.Tensor, labels: torch.Tensor, training: bool = True):
        """
        Computes the feature coverage of a model for a given batch of inputs.

        Args:
            m: Model to evaluate.
            last_batch: the model input for the current step (as a Tensor)
            labels: the model output for the current step (as a Tensor)
            training: boolean - whether the model is training (store codes) or evaluating (determine approximate match accuracy using stored codes)

        Output:
            approximate match accuracy as a fraction.
        """

        # retrieve the hook data
        # the behavior of this hook (return flat list of MACs) is a problem for the use case of this metric
        (
            layers, layer_inputs, layer_outputs
        ) = self.hook.get_layer_io()

        # initialize the result list
        results = []

        # for each input
        for image_index, image_3d in enumerate(last_batch):

            # initialize the cache
            rf_cache = [[] for i in range(len(layers))]

            image = image_3d.flatten()

            image_results = []

            # for each layer
            for layer_index, layer in enumerate(layers):

                # create the empty layerwise input mask
                layer_mask = torch.zeros_like(image, dtype=torch.bool)

                # for each MAC in that layer
                for mac in layer:
                    
                    # create a new input mask in the shape of the input
                    mac_mask = torch.zeros_like(image, dtype=torch.bool)
                    #print(f"MAC {mac_index} Sources: {[x for x, y in enumerate(mac.input_filter) if y != 0]}")

                    # if this MAC is active (=any nonzero neuron outputs = any nonzero values in the 2D tensor for this MAC's output)
                    if mac.is_active:
                        # then update the MAC input mask with this MAC's input
                        if layer_index == 0:
                            # IF we are in layer 1 then the source "MACs" represent pixels in the input that we need to scatter onto the mask
                            mac_mask.scatter_(0, mac.input_filter, 1)
                        else:
                            # otherwise, for each input MAC to this one
                            for source_index in mac.input_filter:
                                # get its input filter from the cache and OR it into the mac_mask
                                # this is slightly inefficient because it does (# of MACs) fetch/OR steps rather than (# of active MACs)
                                # WARNING double check index numbering!
                                mac_mask = torch.bitwise_or(mac_mask, rf_cache[layer_index - 1][source_index])

                        # then OR this MAC into the current layer mask
                        layer_mask = torch.bitwise_or(layer_mask, mac_mask)

                    # regardless, save to the cache as the input mask for this MAC
                    # (this way, MACs that are not active just get an all-zero mask)
                    rf_cache[mac.layer_index].append(mac_mask)

                # when MAC processing has finished

                # AND the layer mask with the input pixels to get the covered pixels
                covered_pixels = torch.logical_and(layer_mask, image)

                # count the number of 1s in the covered pixels
                covered_count = torch.count_nonzero(covered_pixels).item()

                # XOR this with the active pixels to get the uncovered pixels
                uncovered_pixels = torch.logical_xor(covered_pixels, image)

                # count the number of 1s in the uncovered pixels
                uncovered_count = torch.count_nonzero(uncovered_pixels).item()
        
                # total
                total_count = image.count_nonzero().item()

                # at Level 3 verbosity print the uncovered pixel mask
                #print(f"Layer {layer_index}: covered {covered_count} uncovered {uncovered_count}")

                # handle the case of an empty input
                if total_count == 0:
                    # if so we have "covered" all zero features
                    feature_coverage = 1.0
                else:
                    # calculate feature coverage in results as covered pixels / input size
                    feature_coverage = (covered_count / total_count)
                
                # append to results for this item
                image_results.append(feature_coverage)

            # append this item's results
            results.append(image_results) # confirm dimensions are consistent with other metrics

            # if lesser granularity has been requested, average the layerwise results to achieve a single number 
            # (n.b. this probably needs to be weighted by # of macs per layer)

        # return the results
        return results