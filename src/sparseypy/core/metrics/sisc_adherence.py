# -*- coding: utf-8 -*-

"""
SISC Adherence: file holding the SISCAdherenceMetric class.
"""


from typing import Optional, Callable

import torch

from sparseypy.access_objects.models.model import Model
from sparseypy.core.metrics.inputs_and_codes_metric import InputsAndCodesMetric
from sparseypy.core.metrics.comparisons import max_by_layerwise_mean


class SiscAdherenceMetric(InputsAndCodesMetric):
    """
    SISCAdherenceMetric: metric computing the SISC adherence
        of a Sparsey model.
    """
    def __init__(self, model: torch.nn.Module,
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
            model, device, 'sisc_adherence',
            reduction, best_value, approximation_batch_size
        )


    def _compute(self, m: Model, last_batch: torch.Tensor,
                labels: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Computes the code similarity of an input (and associated code)
        with previous inputs and codes.

        Args:
            m (Model): Model to evaluate.
            last_batch (torch.Tensor): the model input for the current step
            labels (torch.Tensor): the model output for the current step
            training (bool): whether the model is training or evaluating

        Returns:
            (torch.Tensor): a list of lists containing the code similarity
                for each MAC in the model.
        """
        _, inputs, outputs = self.hook.get_layer_io()
        batch_size = inputs[0].shape[0]

        if self.stored_codes is None:
            self.initialize_storage(inputs, outputs, batch_size)

        input_images = inputs[0].view(batch_size, -1)

        if torch.sum(self.active_input_slots):
            input_similarities = self.compute_input_similarities(input_images)
            code_similarities = self.compute_code_similarities(outputs)

            metric_values = []

            for layer_similarities in code_similarities:
                metric_values.append(
                    torch.nn.functional.cosine_similarity(
                        layer_similarities,
                        input_similarities,
                        dim=2
                    )
                )
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
