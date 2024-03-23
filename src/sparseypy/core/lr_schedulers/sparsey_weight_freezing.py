# -*- coding: utf-8 -*-

"""
SparseyWeightFreezing: file holding the SparseyWeightFreezing LRScheduler class.
"""


import sys

from typing import Iterator, Callable, List, Optional

import torch

from sparseypy.core.hooks import LayerIOHook


class SparseyWeightFreezing(torch.optim.lr_scheduler):
    def __init__(self, model: torch.nn.Module, opt: torch.optim.Optimizer, thresh = 0.5):
        super().__init__(model.parameters(), dict())
        self.model = model
        self.hook = LayerIOHook(self.model)
        self.opt = opt
        self.thresh = thresh



    def step(self, epoch=None) -> List[torch.Tensor]:
        """
        Returns a list of tensors - one for each MAC in the model - each tensor indicating which weights are frozen for that MAC 
        (0s indicating frozen weights and 1s denoting weights that can be updated)
        """
        weight_freezing = []

        #Use hook to iterate through every layer and get the weights for each layer
            #for each MAC in a layer
                #for each neuron in a mac
                #
        layers, inputs, outputs = self.hook.get_layer_io() #layers is set to each layer of a module

        for layer in layers: #iterate through every layer
            for mac in layer.mac_list: #iterate through every mac in the layer
                weights = mac.weights #grab the weights for that mac

                #calculate the mean of inputs for each neuron in each CM, circle back here and clarify on the return shape
                mean_inputs = torch.mean(weights, dim=1)

                #compare the mean with self.thresh and determine freezing
                #(mean_inputs > self.thresh) creates a Boolean tensor, where true indicates mean exceets threshold
                #.float() converts True/False to 1.0/0.0
                freezing_mask = (mean_inputs > self.thresh).float()

                #prepare the mask with the same shape as weights with 0s for frozen and 1s for updateable
                #since freezing occurs per neuron, we need to expand the freezing_mask to match the shape of the weights
                freezing_mask_expanded = freezing_mask.unsqueeze(1).expand_as(weights)

                #invert the mask (1- mask) since 1s in freezing_mask denote frozen weights, but we need 1s for updateable weights
                updateable_mask = 1 - freezing_mask_expanded

                #add the tensor to the weight_freezing list
                weight_freezing.append(updateable_mask)

                #iterate through each cm and each neuron
                #for each neuron, utilize the mean function and the returned list of input neurons to determine whether it needs to be frozen or nah
            break
        
            
        return weight_freezing
