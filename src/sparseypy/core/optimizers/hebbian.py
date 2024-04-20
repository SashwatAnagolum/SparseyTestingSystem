# -*- coding: utf-8 -*-

"""
Hebbian: file holding the Hebbian optimizer class.
"""


import sys

from typing import Iterator, Callable, Optional

import torch

from sparseypy.core.hooks import LayerIOHook
from sparseypy.core.model_layers.sparsey_layer import MAC


class HebbianOptimizer(torch.optim.Optimizer):
    def __init__(self, model: torch.nn.Module, device: torch.device, 
                 epsilon: float = 1e-7):
        super().__init__(model.parameters(), dict())

        self.model = model
        self.saturation_thresholds = []
        self.timesteps = []
        self.epsilon = epsilon
        self.device = device

        # BUG this will not operate correctly in certain kinds of model
        for layer in model.children():
            if hasattr(layer, 'saturation_threshold'):
                self.saturation_thresholds.append(layer.saturation_threshold)

        self.model_layer_inputs = []
        self.model_layer_outputs = []
        self.model_layers = []

        self.verbosity = 0

        self.hook = LayerIOHook(self.model)

    def calculate_freezing_mask(self, weights, layer_index):
        #Retrieve the threshold for freezing weights for the current layer from a list of thresholds.
        layer_threshold = self.saturation_thresholds[layer_index]
        
        # Calculate the mean of the input weights across dimension 1, which should be inputs.
        # This represents the average activation for each neuron/feature.
        #FRACTION OF WEIGHTS SET TO 1 FOR EACH NEURON
        mean_inputs = torch.mean(weights, dim=1)
        
        #If verbosity is set to a level above 0, print the mean values of the inputs.
        # Converting the tensor to a numpy array for easier reading.
        #if (self.verbosity > 0):
        # print("Mean Inputs: " + str(mean_inputs.numpy()))
        
        #Create a mask where 1 represents weights that are not frozen (updateable)
        # and 0 represents weights that are frozen based on the layer's threshold.
        freezing_mask = (mean_inputs > layer_threshold).float()

        #Expand the freezing mask to match the dimensions of the weights tensor.
        # This is done by adding a new dimension and then expanding it to match the size of weights.
        freezing_mask_expanded = freezing_mask.unsqueeze(1).expand_as(weights)

        #Invert the mask: now 1s represent weights that should be frozen (not updateable),
        # and 0s represent weights that can still be updated.
        updateable_mask = 1 - freezing_mask_expanded

        #If verbosity is set to a level above 0, print the updateable mask.
        # Converting the tensor to a numpy array for easier reading.
        if (self.verbosity > 0):
            print("Updateable Mask: " + str(updateable_mask.numpy()))

        #Return the mask indicating which weights are updateable (not frozen).
        return updateable_mask


    def apply_permanence_update(self, mac: MAC, params: torch.Tensor,
                                layer_index: int, mac_index: int) -> None:
        """
        Applies the permanence weight updates.

        Args:
            mac (MAC): the mac to apply updates to.
            params (torch.Tensor): the weight tensor to update.
            layer_index (int): the layer the MAC is in.
            mac_index (int): the index of the MAC.
        """
        permanence_numerator = torch.pow(
            torch.sub(
                mac.permanence_steps,
                self.timesteps[layer_index][mac_index]
            ).float(), mac.permanence_convexity
        )

        torch.mul(
            params > self.epsilon,
            torch.div(
                permanence_numerator,
                mac.permanence_steps ** mac.permanence_convexity
            ),
            out=params
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

        # Iterate over each layer 
        for layer_index, layer in enumerate(layers):
            if len(self.timesteps) == layer_index:
                self.timesteps.append([])

            # iterate over each MAC along with its input and output tensors.
            for (mac_index, (mac, mac_input, mac_output)) in enumerate(
                zip(layer, inputs[layer_index], outputs[layer_index])
            ):
                if len(self.timesteps[layer_index]) == mac_index:
                    self.timesteps[layer_index].append(
                        torch.zeros(
                            mac.weights.shape, dtype=torch.int32,
                            device=self.device
                        )
                    )

                # iterate over the parameters of the current MAC
                for params in mac.parameters():
                    # Calculate weight updates by performing matrix multiplication between
                    # the transpose of the flattened layer input and the flattened layer output.
                    weight_updates = torch.matmul(
                        torch.transpose(
                            mac_input.view(mac_input.shape[0], -1), 0, 1
                        ),
                        mac_output.view(mac_output.shape[0], -1)
                    )

                    # Reshape the weight updates to match the dimensions of the parameters
                    # and then permute the dimensions for correct alignment.
                    weight_updates = torch.permute(
                        weight_updates.view(
                            weight_updates.shape[0],
                            params.shape[0],
                            params.shape[2]
                        ),
                        (1, 0, 2)
                    )

                    # Print the current parameter values for investigation.
                    # Note: Converting to NumPy for easier visualization.
                    if self.verbosity > 0:
                        print("Params.data " + str(params.data.numpy()))

                    # Calculate the updateable mask based on the current parameters and layer's threshold.
                    updateable_mask = self.calculate_freezing_mask(
                        params.data, layer_index
                    )

                    # Apply the updateable mask to the weight updates, effectively zeroing
                    # updates for weights that are not updateable (frozen).
                    # BUG: probably does not update weights that are both active on this step *and* frozen
                    weight_updates *= updateable_mask

                    # apply permanence/weight decay to all weights 
                    # CHECK whether we need to ignore the frozen weights for decay; if so more will be needed...                
                    self.timesteps[layer_index][mac_index] += 1
                    self.apply_permanence_update(
                        mac, params, layer_index, mac_index
                    )

                    params[torch.ge(weight_updates, 1)] = 1.0
                    self.timesteps[layer_index][mac_index] = torch.mul(
                        torch.lt(weight_updates, 1),
                        self.timesteps[layer_index][mac_index]
                    )

        return
