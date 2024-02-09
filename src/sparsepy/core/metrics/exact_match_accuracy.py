import abc

from typing import Optional

import torch
from sparsepy.access_objects.models.model import Model
from sparsepy.core.model_layers.sparsey_layer import MAC
from sparsepy.core.hooks import LayerIOHook
from sparsepy.core.metrics.metrics import Metric

class ExactMatchAccuracyMetric(Metric):
    def __init__(self, model: torch.nn.Module, reduction: Optional[str] = None):
        super().__init__(model)
        # attaches the hook anew for this Metric to gain access to the hook data
        # consider hook manager later if we need to use many metrics with hooks
        #self.hook = LayerIOHook(self.model, 'highest')
        self.hook = LayerIOHook(self.model, flatten=False)
        # initialize input map
        self.stored_inputs = {}
        self.reduction = reduction

        self.input_images = []

    def get_normalized_hamming_distance(self,
        stored_code: torch.Tensor,
        selected_code: torch.Tensor) -> float:
        """
        Computes the normalized hamming distance between two codes.

        Args:
            stored_code (list[torch.Tensor]): the code generated for a
                particular input during training
            selected_code (list[torch.Tensor]): the code generated for
                the same input during evaluation
        """
        diffs = torch.abs(torch.sub(stored_code, selected_code))

        return torch.mean(torch.lt(diffs, 1e-5).float()).item()


    def compute(self, m: Model, last_batch: torch.Tensor,
                labels: torch.Tensor, training: bool = True):
        """
        Computes the exact match accuracy of a model for a given
        batch of inputs.

        Args:
            m: Model to evaluate.
            last_batch: the model input for the current step (as a Tensor)
            labels: the model output for the current step (as a Tensor)
            training: boolean - whether the model is training (store codes)
                or evaluating (determine exact match accuracy
                using stored codes)

        Output:
            Exact match accuracy as a fraction.
        """
        # fetch the outputs from each model layer using the hook
        # so we can get the codes for the current input item
        (
            layers, layer_inputs, layer_outputs
        ) = self.hook.get_layer_io()

        fidelities = [[] for i in range(len(layers))]

        # loop over batch items
        for image_index, image in enumerate(last_batch):
            # construct dict keys by flattening image tensor to 1D,
            # converting to ints, and concatenating to a string
            image_str = "".join(
                [str(i.item()) for i in image.flatten().int()]
            )

            self.input_images.append(image_str)

            # if we are training, we should store it
            if training:
                self.stored_inputs[image_str] = layer_outputs
            # if we are evaluating, we should 
            # check if we have seen this input before
            # and compare model outputs if we have
            else:
                if image_str in self.stored_inputs.keys():
                    # then we need to determine whether
                    # the codes are the same
                    # for every layer in the output
                    for layer_index in range(len(layer_outputs)):
                        for mac_index in range(len(layer_outputs[layer_index])):
                            fidelities[layer_index].append(
                                self.get_normalized_hamming_distance(
                                    self.stored_inputs[image_str][layer_index][mac_index],
                                    layer_outputs[layer_index][mac_index][image_index]
                                )
                            )

            print(image_str)

        # non-None reductions need updating for the extra added dimension
        if self.reduction is None:
            return fidelities
        elif self.reduction == 'mean':
            return [
                sum(fid_list) / len(fid_list) if len(fid_list) > 0 else None for fid_list in fidelities
            ]
        elif self.reduction == 'sum':
            return [
                sum(fid_list) for fid_list in fidelities
            ]
        else:
            return None
