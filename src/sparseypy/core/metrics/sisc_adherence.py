# -*- coding: utf-8 -*-

"""
SISC Adherence: file holding the SISCAdherenceMetric class.
"""


import random
from typing import Optional, Callable

import torch

from sparseypy.access_objects.models.model import Model
from sparseypy.core.hooks import LayerIOHook
from sparseypy.core.metrics.metrics import Metric
from sparseypy.core.metrics.comparisons import max_by_layerwise_mean


class SiscAdherenceMetric(Metric):
    """
    SISCAdherenceMetric: metric computing the SISC adherence
        of a Sparsey model.

    Attributes:
        reduction (str): the type of reduction to apply
            onto the raw per-layer, per-sample feature coverage
            results. Valid options are None and 'sparse'. Choosing
            'sparse' will return the raw averaged inputs to each MAC.
            Choosing None will return the inputs inserted into
            their positions in a tensor of the same size as the 
            input samples to the model.
        hook (LayerIOHook): the hook registered with the model
            being evaluated to obtain references to each layer,
            and layerwise inputs and outputs.
    """
    def __init__(self, model: torch.nn.Module,
                 device: torch.device,
                 reduction: Optional[str] = None,
                 best_value: Optional[Callable] = max_by_layerwise_mean) -> None:
        """
        Initializes the SiscAdherenceMetric object. 

        Args:
            model (torch.nn.Module): the model to compute feature
                coverage using.
            reduction (Optional[str]): the type of reduction
                to apply before returning the metric value.
        """
        super().__init__(
            model, "sisc_adherence",
            best_value, device, reduction
        )

        self.hook = LayerIOHook(self.model)
        self.approximation_batch_size = 64
        self.stored_codes = None
        self.stored_inputs = None
        self.active_input_slots = None


    def compute_similarity_correlation(self, images: torch.Tensor,
                           outputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Compute the code similarity between the current code and 
        previously stored codes.
        """
        correlations = []

        if not torch.sum(self.active_input_slots):
            return [
                [0.0 for i in range(output.shape[1])]
                for output in outputs
            ]

        numerator = torch.logical_and(self.stored_inputs, images).sum(2)
        denominator = torch.logical_or(self.stored_inputs, images).sum(2)
        image_similarities = torch.div(numerator, denominator).transpose(
            0, 1
        ).unsqueeze(1)

        for layer_index, output in enumerate(outputs):
            code_sim_num = torch.logical_and(
                self.stored_codes[layer_index],
                output
            ).sum(3)

            code_sim_denom = torch.logical_or(
                self.stored_codes[layer_index],
                output
            ).sum(3)

            layer_similarities = torch.div(
                code_sim_num, code_sim_denom
            ).permute(1, 2, 0)

            torch.nan_to_num(
                layer_similarities, 0.0, out=layer_similarities
            )

            layer_correlations = torch.nn.functional.cosine_similarity(
                layer_similarities[:, :, self.active_input_slots],
                image_similarities[:, :, self.active_input_slots], dim=2
            )

            correlations.append(torch.mean(layer_correlations, dim=0))

        return correlations


    def update_stored_images_and_codes(self,
        images: torch.Tensor, outputs: list[torch.Tensor],
        batch_size: int) -> None:
        """
        Update the list of stored images and codes using images
        and model outputs from the current batch.
        """
        num_images_to_swap = torch.randint(
            0, min(self.approximation_batch_size, batch_size), (1,)
        ).item()

        if num_images_to_swap:
            swap_indices = torch.randint(
                0, self.approximation_batch_size,
                (num_images_to_swap,)
            )

            select_indices = torch.randint(
                0, batch_size, (num_images_to_swap,)
            )
            
            for layer_index, output in enumerate(outputs):
                self.stored_codes[layer_index][
                    swap_indices
                ] = output[select_indices].unsqueeze(1)

            self.stored_inputs[swap_indices] = images[
                select_indices
            ].unsqueeze(1)

            self.active_input_slots[swap_indices] = True


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
            (list[list[float]]): a list of lists containing the code similarity
                for each MAC in the model.
        """
        _, inputs, outputs = self.hook.get_layer_io()
        batch_size = inputs[0].shape[0]

        if self.stored_codes is None:
            self.stored_codes = []
            self.stored_inputs = torch.zeros(
                (
                    self.approximation_batch_size, 1,
                    inputs[0].numel() // batch_size
                ),
                dtype=torch.float32,
                device=self.device
            )

            self.active_input_slots = torch.zeros(
                self.approximation_batch_size,
                dtype=torch.bool, device=self.device
            )

            self.stored_codes = [
                torch.zeros(
                    self.approximation_batch_size, 1,
                    output.shape[1], output.shape[2]
                ) for output in outputs
            ]

        input_images = inputs[0].squeeze_(2)
        metric_values = self.compute_similarity_correlation(
            input_images, outputs
        )

        if training:
            self.update_stored_images_and_codes(
                input_images, outputs, batch_size
            )

        return torch.nested.nested_tensor(
            metric_values, dtype=torch.float32, device=self.device
        )
