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
    def __init__(self) -> None:
        """
        Initializes the model. 
        """
        super().__init__()

        self.layers = torch.nn.ModuleList()


    def add_layer(self, layer: torch.nn.Module) -> None:
        """
        Adds a layer to the layers list of the model.
        """
        self.layers.append(layer)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass with data x.

        Args:
            x (torch.Tensor): the data to pass through the model.

        Returns:
            (torch.Tensor): the output of the model.
        """
        for layer in self.layers:
            x = layer(x)

        return x
