# -*- coding: utf-8 -*-

"""
Model: file defining the Model class.
"""


import torch


class Model(torch.nn.Module):
    """
    Model: a class to represent model objects used by the system.

    Attributes:
        layers: the layers in the model.
    """
    def __init__(self, device: torch.device) -> None:
        """
        Initializes the model. 
        """
        super().__init__()

        self.num_layers = 0
        self.device = device


    def add_layer(self, layer: torch.nn.Module) -> None:
        """
        Adds a layer to the layers list of the model.
        """
        self.add_module(f'Layer_{self.num_layers}', layer)

        self.num_layers += 1


    def train(self, mode: bool = True) -> None:
        """
        Sets the model to training mode.
        Args:
            mode (bool): whether to set the model to training mode.
        """
        for module in self.children():
            module.train(mode)               


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass with data x.

        Args:
            x (torch.Tensor): the data to pass through the model.

        Returns:
            (torch.Tensor): the output of the model.
        """
        for layer_num in range(self.num_layers):
            x = self.get_submodule(f'Layer_{layer_num}')(x)

        return x
