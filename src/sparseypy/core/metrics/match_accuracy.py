# -*- coding: utf-8 -*-

"""
Match Accuracy: file holding the MatchAccuracyMetric class.
"""


from typing import Optional
import torch

from typing import Optional, Callable

from sparseypy.access_objects.models.model import Model
from sparseypy.core.metrics.inputs_and_codes_metric import InputsAndCodesMetric
from sparseypy.core.metrics.comparisons import max_by_layerwise_mean


class MatchAccuracyMetric(InputsAndCodesMetric):
    """
    MatchAccuracyMetric: class computing the match
        accuracy for a Sparsey model over a batch of input
        images.
    """
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 reduction: Optional[str] = None,
                 best_value: Optional[Callable] = max_by_layerwise_mean,
                 approximation_batch_size: Optional[int] = 64) -> None:
        """
        Initializes the SiscAdherenceMetric object. 

        Args:
            model (torch.nn.Module): the model to compute feature
                coverage using.
            device (torch.device): the device to use.
            reduction (Optional[str]): the type of reduction
                to apply before returning the metric value.
            best_value (Callable): the comparison function
                to use to determine the best value obtained for this
                metric.
            approximation_batch_size (Optional[int]): the 
                size of the approximation batch to use while
                computing the metric.
        """
        super().__init__(
            model, device, 'match_accuracy',
            reduction, best_value, approximation_batch_size
        )


    def _compute(self, m: Model, last_batch: torch.Tensor,
                 labels: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Computes the approximate match accuracy of a model
        for a given batch of inputs.

        Args:
            m: Model to evaluate.
            last_batch: the model input for the current step (as a Tensor)
            labels: the model output for the current step (as a Tensor)
            training: boolean - whether the model is training (store codes)
                or evaluating (determine approximate match
                accuracy using stored codes)

        Output:
            approximate match accuracy as a list of accuracies:
            one pertaining to each layer
        """
        _, inputs, outputs = self.hook.get_layer_io()
        batch_size = inputs[0].shape[0]
        input_images = inputs[0].view(batch_size, -1)

        if self.stored_codes is None:
            self.initialize_storage(inputs, outputs, batch_size)

        if not training and torch.sum(self.active_input_slots):
            input_similarities = self.compute_input_similarities(input_images)
            input_similarities = input_similarities.squeeze(1)
            closest_code_indices = torch.argmax(input_similarities, 1)

            metric_values = []

            for layer_index, output in enumerate(outputs):
                closest_codes = self.stored_codes[layer_index][
                    closest_code_indices
                ].squeeze(1)

                sim_num = torch.logical_and(closest_codes, output).sum(2)
                sim_den = torch.logical_or(closest_codes, output).sum(2)
                layer_sims = torch.div(sim_num, sim_den)
                torch.nan_to_num(layer_sims, 0.0, out=layer_sims)

                metric_values.append(layer_sims)
        else:
            metric_values = [
                torch.zeros(
                    (batch_size, output.shape[1]),
                    dtype=torch.float32,
                    device=self.device
                )
                for output in outputs
            ]

        if training:
            self.update_stored_images_and_codes(
                input_images, outputs, batch_size
            )

        return torch.nested.nested_tensor(
            metric_values, dtype=torch.float32, device=self.device
        )
