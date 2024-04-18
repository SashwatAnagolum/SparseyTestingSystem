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
        super().__init__(model, "sisc_adherence", best_value, device)

        self.reduction = reduction
        self.hook = LayerIOHook(self.model)
        self.approximation_batch_size = 64
        self.stored_inputs = []
        self.stored_codes = []


    def compute_image_similarity(self, tensor_1: torch.Tensor,
                                  tensor_2: torch.Tensor) -> float:
        """
        Returns the similarity between two tensors of the same shape.
        """
        return 1.0 - (
                torch.sum(
                torch.logical_xor(
                    tensor_1, tensor_2
                )
            ) / tensor_1.numel()
        ).item()


    def compute_code_similarity(self, tensor_1: torch.Tensor,
                                  tensor_2: torch.Tensor) -> float:
        """
        Returns the similarity between two tensors of the same shape.
        """
        return (
            torch.nan_to_num(
                    torch.sum(
                    torch.logical_and(tensor_1, tensor_2)
                ) / (0.5 * torch.sum(tensor_1) + torch.sum(tensor_2)),
                1.0
            )
        ).item()


    def compute_similarity(self,
                                image: torch.Tensor,
                                code: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Compute the code similarity between the current code and 
        previously stored codes.
        """
        image_similarities = []
        code_similarities = []

        for stored_input in self.stored_inputs:
            image_similarities.append(
                self.compute_image_similarity(
                    stored_input, image
                )
            )

        for layer_index, layer_code in enumerate(code):
            code_similarities.append([])

            for mac_index, mac_code in enumerate(layer_code):
                code_similarities[-1].append([])

                for stored_code in self.stored_codes:
                    code_similarities[-1][-1].append(
                        self.compute_code_similarity(
                            stored_code[layer_index][mac_index],
                            mac_code
                        )
                    )

        code_similarities = [
            [torch.Tensor(mac_sim) for mac_sim in layer_sim]
            for layer_sim in code_similarities
        ]

        image_similarities = torch.Tensor(image_similarities)

        similarities = [
            [
                torch.nn.functional.cosine_similarity(
                    code_similarities[layer_index][mac_index],
                    image_similarities, dim=0
                ).item() for mac_index in range(len(code_similarities[layer_index]))
            ] for layer_index in range(len(code_similarities))
        ]

        if self.reduction is None or self.reduction == 'none':
            return similarities
        elif self.reduction == 'layerwise_mean':
            return [
                sum(layer_similarities) / len(layer_similarities)
                for layer_similarities in similarities
            ]
        elif self.reduction == 'sum':
            return sum([sum(layer_sim) for layer_sim in similarities])
        elif self.reduction == 'mean':
            return (
                sum([sum(layer_sim) for layer_sim in similarities]) / 
                sum([len(layer_sim) for layer_sim in similarities])
            )
        elif self.reduction == 'highest_layer':
            return similarities[-1]
        else:
            return None


    def compute(self, m: Model, last_batch: torch.Tensor,
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
        _, _, layer_outputs = self.hook.get_layer_io()

        for image in last_batch:
            if len(self.stored_inputs) == 0:
                code_similarity = [
                    [0.0 for i in range(len(layer_outputs[idx]))]
                    for idx in range(len(layer_outputs))
                ]
            else:
                code_similarity = self.compute_similarity(image, layer_outputs)

            if len(self.stored_inputs) < self.approximation_batch_size:
                self.stored_inputs.append(image)
                self.stored_codes.append(layer_outputs)
            else:
                index = random.randint(0, self.approximation_batch_size - 1)
                self.stored_codes[index] = layer_outputs
                self.stored_inputs[index] = image

            return code_similarity
