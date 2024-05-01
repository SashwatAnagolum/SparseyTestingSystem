# -*- coding: utf-8 -*-

"""
Hebbian: file holding the Hebbian optimizer class.
"""


import sys
import torch

from sparseypy.core.hooks import LayerIOHook
from sparseypy.core.model_layers.sparsey_layer import MAC


class HebbianOptimizer(torch.optim.Optimizer):
    """
    HebbianOptimizer: class representing the optimizer
        for the Sparsey model.
        Attributes:
            model (torch.nn.Module): the model to optimize.
            device (torch.device): the device to run the model on.
            epsilon (float): the epsilon value to use for
                numerical stability.
            saturation_thresholds (list[float]): the saturation
                thresholds for each layer.
            timesteps (dict): the number of timesteps that each
                weight has not been updated for.
            verbosity (int): the verbosity level.
            hook (LayerIOHook): the hook to use for
                retrieving layer inputs and outputs.
    """
    def __init__(self, model: torch.nn.Module, device: torch.device, 
                 epsilon: float = 1e-7):
        """
        Initialize the HebbianOptimizer.
        Args:
            model (torch.nn.Module): the model to optimize.
            device (torch.device): the device to run the model on.
            epsilon (float): the epsilon value to use for
                numerical stability.
        """
        super().__init__(model.parameters(), dict())

        self.model = model
        self.saturation_thresholds = []
        self.timesteps = dict()
        self.epsilon = epsilon
        self.device = device
        self.verbosity = 0
        self.hook = LayerIOHook(self.model)

        for layer in model.children():
            if hasattr(layer, 'saturation_threshold'):
                self.saturation_thresholds.append(layer.saturation_threshold)
            else:
                self.saturation_thresholds.append(1.0)


    def calculate_freezing_mask(self, weights, layer_index):
        """
        Calculates the freezing mask for the weights of a SparseyLayer.
        """
        active_weights_frac = torch.mean(weights, dim=1, keepdim=True)
        weight_update_mask = torch.gt(
            active_weights_frac, self.saturation_thresholds[layer_index]
        ).expand_as(weights)

        return weight_update_mask


    def apply_permanence_update(self, permanence_steps: int,
                                permanence_convexity: float,
                                params: torch.Tensor,
                                timestep_values: torch.Tensor) -> None:
        """
        Applies the permanence weight updates.

        Args:
            permanence_steps (int): the number of steps before
                unupdated weights are reduced to zero.
            permanence_convexity (float): controls the slope
                of the function used to reduce the values of weights
                that are not updated.
            params (torch.Tensor): the weight tensor to update.
            timestep_values (torch.Tensor): the timesteps that
                each weight in params has not been updated for.
        """
        torch.div(
            1.0 + (permanence_convexity / permanence_steps),
            torch.add(
                torch.div(
                    permanence_convexity,
                    torch.sub(
                        permanence_steps,
                        timestep_values
                    )
                ), 1.0
            ),
            out=params
        )

        torch.where(
            torch.ge(
                timestep_values,
                permanence_steps
            ), torch.zeros(1, device=self.device),
            params, out=params
        )


    def step(self, closure=None) -> None:
        """
        Performs a weight update.

        Args:
            closure: callable returning the model output.
        """
        # Retrieve layers, their inputs, and outputs using the custom hook.
        # 'layers' contains instances of MAC,
        # 'inputs' and 'outputs' are tensors repres
        # enting inputs and outputs for those MACs.
        layers, inputs, outputs = self.hook.get_layer_io()

        with torch.no_grad():
            # Iterate over each layer
            for layer_index, (layer, layer_input, layer_output) in enumerate(
                zip(layers, inputs, outputs)
            ):
                if layer_index not in self.timesteps:
                    self.timesteps[layer_index] = []

                for param_index, params in enumerate(layer.parameters()):
                    if len(self.timesteps[layer_index]) == param_index:
                        self.timesteps[layer_index] = torch.ones(
                            params.shape, dtype=torch.float32,
                            device=self.device
                        )

                        torch.mul(
                            self.timesteps[layer_index],
                            layer.permanence_steps,
                            out=self.timesteps[layer_index]
                        )

                    layer_input = torch.cat(
                        (
                            layer_input,
                            torch.zeros(
                                (
                                    layer_input.shape[0],
                                    1, *layer_input.shape[2:]
                                ),
                                dtype=torch.float32, device=self.device
                            )
                        ), dim=1
                    )

                    mac_inputs = layer_input[:, layer.input_connections]

                    weight_updates = torch.matmul(
                        torch.permute(
                            mac_inputs.view(*mac_inputs.shape[:2], -1),
                            (1, 2, 0)
                        ),
                        torch.permute(layer_output, (1, 0, 2))
                    )

                    weight_freeze_mask = self.calculate_freezing_mask(
                        params, layer_index
                    )

                    torch.div(
                        weight_updates,
                        layer_input.shape[0],
                        out=weight_updates
                    )

                    weight_updates[weight_freeze_mask] = 0.0

                    torch.add(
                        self.timesteps[layer_index], 1,
                        out=self.timesteps[layer_index]
                    )

                    self.apply_permanence_update(
                        layer.permanence_steps,
                        layer.permanence_convexity,
                        params, self.timesteps[layer_index]
                    )

                    torch.add(params, weight_updates, out=params)
                    torch.clamp(params, 0.0, 1.0, out=params)
                    self.timesteps[layer_index][
                        torch.gt(weight_updates, 0)
                    ] = 0

        return
