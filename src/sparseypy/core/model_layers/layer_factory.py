# -*- coding: utf-8 -*-

"""
Layer Factory: file holding the Layer Factory class.
"""


import torch

from sparseypy.core import model_layers


class LayerFactory:
    """
    Layer Factory: class to create layers based on the layer name.
    attributes:
        allowed_modules (set): set of allowed modules to create layers from.
    """
    allowed_modules = set([i for i in dir(model_layers) if i[:2] != '__'])

    @staticmethod
    def get_layer_class(layer_name):
        """
        Gets the class corresponding to the name passed in.
        Throws an error if the name is not valid.
        """
        class_name = ''.join(
            [l.capitalize() for l in layer_name.split('_')] + ['Layer']
        )

        if class_name in LayerFactory.allowed_modules:
            return getattr(model_layers, class_name)
        elif layer_name in dir(torch.nn):
            return getattr(torch.nn, layer_name)
        else:
            raise ValueError('Invalid layer name!')
    

    @staticmethod
    def create_layer(layer_name, **kwargs) -> torch.nn.Module:
        """
        Creates a layer passed in based on the layer name and kwargs.
        """
        layer_class = LayerFactory.get_layer_class(layer_name)

        layer_obj = layer_class(**kwargs)

        return layer_obj
