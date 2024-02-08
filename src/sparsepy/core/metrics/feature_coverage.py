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
            rf_cache = []

            image = image_3d.flatten()

            image_results = []

            # count MACs to track position in hook
            # (because it is a flat structure we need to know how many MACs we have seen across layers so far to know where the next MAC in this layer is)
            macs_counted = 0

            # for each layer
            for layer_index, layer in enumerate(m.named_children()):

                print(f"Layer {layer_index}")

                rf_cache.append([])

                # create the empty layerwise input mask
                layer_mask = torch.zeros_like(image, dtype=torch.bool)

                # for each MAC in that layer
                for mac_index, mac in enumerate(layer[1].mac_list):
                    
                    # create a new input mask in the shape of the input
                    mac_mask = torch.zeros_like(image, dtype=torch.bool)

                    # get all of the MACs in the previous layer from which this MAC receives input
                    source_macs = mac.input_filter

                    #print(f"MAC {mac_index} Sources: {[x for x, y in enumerate(source_macs) if y != 0]}")
                    
                    # TODO only consider MACs that are currently active as part of the mask building

                    # IF we are in layer 1 then the source "MACs" represent pixels in the input that we need to scatter onto the mask
                    if layer_index == 0:
                        mac_mask.scatter_(0, source_macs, 1)
                        #print(f"Mask shape: {mac_mask.shape}")
                        #print(f"{mac_mask}")

                    # ELSE loop over all the source MACs
                    else:
                    # for each of those source MACs
                        for source_index in source_macs:
                            # get its input filter from the cache and OR it into the mac_mask
                            # WARNING double check index numbering!
                            mac_mask = torch.bitwise_or(mac_mask, rf_cache[layer_index - 1][source_index])
        
                    # save to the cache as the input mask of this MAC
                    # TODO for MACs that are NOT active append a blank mask to the cache
                    rf_cache[layer_index].append(mac_mask)
                    #rf_cache[layer_index][mac_index] = mac_mask
                    #print(mac_mask)
                    # IF this MAC is active
                    if layer_outputs[macs_counted][image_index].any()  > 0:
                        # then OR its finished input mask into this layer's input mask
                        layer_mask = torch.bitwise_or(layer_mask, mac_mask)
                    # regardless of activity, increment MAC counter
                    macs_counted += 1

                # when MAC processing has finished

                # AND the layer mask with the input pixels to get the covered pixels
                covered_pixels = torch.logical_and(layer_mask, image)

                # count the number of 1s in the covered pixels
                covered_count = torch.count_nonzero(covered_pixels)

                # XOR this with the active pixels to get the uncovered pixels
                uncovered_pixels = torch.logical_xor(covered_pixels, image)

                # count the number of 1s in the uncovered pixels
                uncovered_count = torch.count_nonzero(uncovered_pixels)
        
                # at Level 3 verbosity print the uncovered pixel mask
                print(f"Layer {layer_index}: covered {covered_count} uncovered {uncovered_count}")

                # calculate feature coverage in results as covered pixels / input size
                # WARNING unsafe division
                feature_coverage = (covered_count / image.count_nonzero()).item()
                image_results.append(feature_coverage)

            results.append(image_results) # confirm dimensions are consistent with other metrics

            # if lesser granularity has been requested, average the layerwise results to achieve a single number 
            # (n.b. this probably needs to be weighted by # of macs per layer)

        # return the results
        return results