# -*- coding: utf-8 -*-

"""
Model Builder: code for the Model Builder class.
"""


from sparsepy.core.model_layers.layer_factory import LayerFactory
from sparsepy.access_objects.models.model import Model
from sparsepy.core.hooks.hook_factory import HookFactory


class ModelBuilder:
    """
    Model Builder: class to build Model objects.
    """
    @staticmethod
    def build_model(model_config: dict):
        """
        Builds the model layer by layer.

        Args:
            model_config (dict): information
                about the structure of the model and its layers.

        Returns:
            (torch.nn.Module): a Model object that can be trained.
        """
        model = Model()

        for layer_config in model_config['layers']:
            new_layer = LayerFactory.create_layer(
                layer_config['name'], **layer_config['params']
            )

            model.add_layer(new_layer)

        for hook_config in model_config['hooks']:
            hook = HookFactory.create_hook(hook_config['name'], model)

        return model