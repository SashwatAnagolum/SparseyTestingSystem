# -*- coding: utf-8 -*-

"""
Hebbian: file holding the Hebbian optimizer class.
"""


import sys

from typing import Iterator, Callable, Optional

import torch

from sparsepy.core.hooks import LayerIOHook


class HebbianOptimizer(torch.optim.Optimizer):
    def __init__(self, model: torch.nn.Module):
        super().__init__(model.parameters(), dict())
        self.model = model
        self.saturation_thresholds = []
        for layer in model.children():
            if hasattr(layer, 'saturation_threshold'):
                self.saturation_thresholds.append(layer.saturation_threshold)
        self.model_layer_inputs = []
        self.model_layer_outputs = []
        self.model_layers = []

        self.hook = LayerIOHook(self.model)

    def calculate_freezing_mask(self, weights, layer_index):
        layer_threshold = self.saturation_thresholds[layer_index]
        mean_inputs = torch.mean(weights, dim=1)
        freezing_mask = (mean_inputs > layer_threshold).float()
        freezing_mask_expanded = freezing_mask.unsqueeze(1).expand_as(weights)
        updateable_mask = 1 - freezing_mask_expanded
        return updateable_mask

    def step(self, closure=None) -> None:
        """
        Performs a weight update.

        Args:
            closure: callable returning the model output.
        """
        layers, inputs, outputs = self.hook.get_layer_io()

        for (layer, layer_input, layer_output) in zip(
            layers, inputs, outputs
        ):
            for params in layer.parameters():
                weight_updates = torch.matmul(
                    torch.transpose(
                        layer_input.view(layer_input.shape[0], -1), 0, 1
                    ),
                    layer_output.view(layer_output.shape[0], -1)
                )

                weight_updates = torch.permute(
                    weight_updates.view(
                        weight_updates.shape[0],
                        params.shape[0],
                        params.shape[2]
                    ),
                    (1, 0, 2)
                )

                layer_index = layer.layer_index

                updateable_mask = self.calculate_freezing_mask(params.data, layer_index)
                weight_updates *= updateable_mask

                params += torch.ge(weight_updates, 1)
                torch.clamp(params, 0, 1, out=params)

        return
