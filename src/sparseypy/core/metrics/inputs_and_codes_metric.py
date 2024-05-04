# -*- coding: utf-8 -*-

"""
Inputs And Codes Metric: file holding the InputsAndCodesMetric class.
"""


import abc
from typing import Optional, Callable

import torch

from sparseypy.access_objects.models.model import Model
from sparseypy.core.hooks import LayerIOHook
from sparseypy.core.metrics.metrics import Metric
from sparseypy.core.metrics.comparisons import max_by_layerwise_mean


class InputsAndCodesMetric(Metric):
    """
    InputsAndCodesMetric: utility class holding
        methods commonly used for metrics using 
        previously seen inputs and codes.

    Attributes:
    """
    def __init__(self, model: torch.nn.Module,
                 device: torch.device,
                 name: str,
                 reduction: Optional[str] = None,
                 best_value: Optional[Callable] = max_by_layerwise_mean,
                 approximation_batch_size: Optional[int] = 64) -> None:
        """
        Initializes the InputsAndCodesMetric object. 

        Args:
            model (torch.nn.Module): the model to compute feature
                coverage using.
            device (torch.device): the device to use.
            reduction (Optional[str]): the type of reduction
                to apply before returning the metric value.
            best_value (Callable): the comparison function
                to use to determine the best value obtained for this
                metric.
            name (str): the name of the metric being created.
            approximation_batch_size (Optional[int]): the 
                size of the approximation batch to use while
                computing the metric.
        """
        super().__init__(
            model, name,
            best_value, device, reduction
        )

        self.hook = LayerIOHook(self.model)
        self.approximation_batch_size = approximation_batch_size
        self.stored_codes = None
        self.stored_inputs = None
        self.active_input_slots = None
        self.num_active_input_slots = None


    def initialize_storage(self, inputs: torch.Tensor,
                           outputs: torch.Tensor, batch_size: int) -> None:
        """
        Initializes the code and input storage for the metric.
        """
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

        self.num_active_input_slots = 0
        self.stored_codes = [
            torch.zeros(
                self.approximation_batch_size, 1,
                output.shape[1], output.shape[2],
                device=self.device
            ) for output in outputs
        ]


    def compute_input_similarities(self,
                                   inputs: torch.Tensor) -> torch.Tensor:
        """
        Computes the similarities between the inputs to the model
        and the stored inputs in the metric.

        Args:
            inputs (torch.Tensor): the inputs to the model

        Returns:
            (torch.Tensor): similarities between the model
                inputs and all of the stored inputs in the metric.
        """
        numerator = torch.logical_and(
            self.stored_inputs[self.active_input_slots], inputs
        ).sum(2)

        denominator = torch.logical_or(
            self.stored_inputs[self.active_input_slots], inputs
        ).sum(2)

        input_similarities = torch.div(numerator, denominator)
        torch.nan_to_num(input_similarities, 0.0, out=input_similarities)
        input_similarities = input_similarities.transpose(0, 1).unsqueeze(1)

        return input_similarities


    def compute_code_similarities(self,
                                  outputs: torch.Tensor) -> list[torch.Tensor]:
        """
        Computes the similarities between the codes in the output of the model
        and the stored codes in the metric.

        Args:
            outputs (torch.Tensor): the outputs from the model

        Returns:
            (list[torch.Tensor]): similarities between the model
                outputs and all of the stored codes in the metric.
        """
        layer_similarities = []

        for layer_index, output in enumerate(outputs):
            code_sim_num = torch.logical_and(
                self.stored_codes[layer_index][self.active_input_slots],
                output
            ).sum(3)

            code_sim_denom = torch.logical_or(
                self.stored_codes[layer_index][self.active_input_slots],
                output
            ).sum(3)

            similarities = torch.div(
                code_sim_num, code_sim_denom
            ).permute(1, 2, 0)

            torch.nan_to_num(similarities, 0.0, out=similarities)
            layer_similarities.append(similarities)

        return layer_similarities


    def update_stored_images_and_codes(self,
        images: torch.Tensor, outputs: list[torch.Tensor],
        batch_size: int) -> None:
        """
        Update the list of stored images and codes using images
        and model outputs from the current batch.
        """
        random_insertion = False
        num_empty_slots = (
            self.approximation_batch_size - self.num_active_input_slots
        )

        if num_empty_slots >= batch_size:
            num_images_to_swap = batch_size
        elif num_empty_slots > 0:
            num_images_to_swap = num_empty_slots
        else:
            random_insertion = True
            num_images_to_swap = torch.randint(
                0, batch_size + 1, (1,)
            ).item()

        if num_images_to_swap:
            if random_insertion:
                swap_indices = torch.randint(
                    0, self.approximation_batch_size,
                    (num_images_to_swap,),
                    device=self.device
                )

                select_indices = torch.randint(
                    0, batch_size, (num_images_to_swap,),
                    device=self.device
                )
            else:
                select_indices = [i for i in range(num_images_to_swap)]
                swap_indices = [
                    i for i in range(
                        self.num_active_input_slots,
                        self.num_active_input_slots + num_images_to_swap
                    )
                ]

            for layer_index, output in enumerate(outputs):
                self.stored_codes[layer_index][
                    swap_indices
                ] = output[select_indices].unsqueeze(1)

            self.stored_inputs[swap_indices] = images[
                select_indices
            ].unsqueeze(1)

            self.active_input_slots[swap_indices] = True
            self.num_active_input_slots += num_images_to_swap


    @abc.abstractmethod
    def _compute(self, m: Model, last_batch: torch.Tensor,
                labels: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Computes the metric using the previously seen inputs and codes, 
        as well as the current batch of inputs and codes.

        Args:
            m (Model): Model to evaluate.
            last_batch (torch.Tensor): the model input for the current step
            labels (torch.Tensor): the model output for the current step
            training (bool): whether the model is training or evaluating

        Returns:
            (torch.Tensor): the raw metric values
        """
