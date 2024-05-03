# -*- coding: utf-8 -*-

"""
Model Builder: code for the Model Builder class.
"""


import torch

from sparseypy.access_objects.models.model import Model
from sparseypy.core.hooks.hook_factory import HookFactory
from sparseypy.core.model_layers.layer_factory import LayerFactory


class ModelBuilder:
    """
    Model Builder: class to build Model objects.
    """
    @staticmethod
    def build_model(model_config: dict, device: torch.device):
        """
        Builds the model layer by layer.

        Args:
            model_config (dict): information
                about the structure of the model and its layers.

        Returns:
            (torch.nn.Module): a Model object that can be trained.
        """
        model = Model(device)

        for (layer_index, layer_config) in enumerate(model_config['layers']):
            layer_config['params']['layer_index'] = layer_index

            new_layer = LayerFactory.create_layer(
                layer_config['name'], **layer_config['params'], device=device
            )

            model.add_layer(new_layer)

        if 'hooks' in model_config:
            for hook_config in model_config['hooks']:
                hook = HookFactory.create_hook(hook_config['name'], model)

        return model


    @staticmethod
    def rehydrate_model(model_config: dict, state_dict: str, device: torch.device):
        """
        Builds a model layer by layer and then reloads its state
        from an existing state dictionary at the provided path.

        Args:
            model_config (str): information about the structure
                of the model and its layers.
            state_dict (str): the model state dictionary to reload.
            device (torch.device): the device (CPU, CUDA, or MPS) to use.
        """
        model = ModelBuilder.build_model(model_config, device=device)
        model.load_state_dict(state_dict)

        return model
