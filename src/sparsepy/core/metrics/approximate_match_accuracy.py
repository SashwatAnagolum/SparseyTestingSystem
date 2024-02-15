import abc
from typing import Optional
import torch

from typing import Optional

from sparsepy.access_objects.models.model import Model
from sparsepy.core.model_layers.sparsey_layer import MAC
from sparsepy.core.hooks import LayerIOHook
from sparsepy.core.metrics.metrics import Metric

class ApproximateMatchAccuracyMetric(Metric):

    def __init__(self, model: torch.nn.Module, reduction: Optional[str] = None):
    def __init__(self, model: torch.nn.Module, reduction: Optional[str] = None):
        super().__init__(model)
        # attaches the hook anew for this Metric to gain access to the hook data
        # consider hook managerlater if we need to use many metrics with hooks
        self.hook = LayerIOHook(self.model)
        # initialize input map
        self.stored_inputs = {}
        self.input_images = []
        self.reduction = reduction

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

    def compute(self, m: Model, last_batch: torch.Tensor, output: torch.Tensor, training: bool = True):
        """
        Computes the approximate match accuracy of a model for a given batch of inputs.

        Args:
            m: Model to evaluate.
            last_batch: the model input for the current step (as a Tensor)
            labels: the model output for the current step (as a Tensor)
            training: boolean - whether the model is training (store codes) or evaluating (determine approximate match accuracy using stored codes)

        Output:
            approximate match accuracy as a list of accuracies: one pertaining to each layer
        """
        
        # fetch the outputs from each model layer using the hook so we can get the codes for the current input item
        layers, layer_inputs, layer_outputs = self.hook.get_layer_io()

        fidelities = [[] for i in range(len(layers))]

        # loop over batch items
        for image_index, image in enumerate(last_batch):
            
            # construct dict keys by flattening image tensor to 1D,
            # converting to ints, and concatenating to a string
            image_str = "".join(
                [str(i.item()) for i in image.flatten().int()]
            )

            if training:
                self.stored_inputs[image_str] = layer_outputs
                self.input_images.append(image_str)

            else:
                # Find a similar image string in the stored inputs using the approximation_tolerace as a threshold for tensor similarity
                similar_image_str = None

                # Initialize a variable to track the minimum number of differing bits
                min_diff_bits = float('inf')
                similar_image_str = None

                # Convert stored inputs to binary tensors and perform XOR and sum operations
                #May require we use a version that isnt condensed here
                for stored_image_str in self.stored_inputs.keys():
                    stored_image_tensor = torch.tensor([int(i) for i in stored_image_str], dtype=torch.int64).view(image.shape)
                    # Perform XOR operation between the binary image and stored binary images
                    diff = torch.bitwise_xor(image.to(torch.int64), stored_image_tensor)
                    # Sum the bits - count of 1's will give the number of differing bits
                    diff_sum = torch.sum(diff).item()

                    # Find the position with the lowest sum of differing bits
                    if diff_sum < min_diff_bits:
                        min_diff_bits = diff_sum
                        similar_image_str = stored_image_str
                
                #similar_image_output = self.stored_inputs[similar_image_str]

                for layer_index in range(len(layer_outputs)):
                        for mac_index in range(len(layer_outputs[layer_index])):
                            fidelities[layer_index].append(
                                self.get_normalized_hamming_distance(
                                    self.stored_inputs[similar_image_str][layer_index][mac_index],
                                    layer_outputs[layer_index][mac_index][image_index]
                                )
                            )
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
