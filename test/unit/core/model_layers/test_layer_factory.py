# -*- coding: utf-8 -*-

"""
Test Layer Factory: test cases for the LayerFactory class.
"""


import pytest
import torch

from sparsepy.core import model_layers
from sparsepy.core.model_layers.layer_factory import LayerFactory


class TestLayerFactory:
    """
    TestLayerFactory: a class holding a collection
        of tests focused on the LayerFactory class.
    """
    def test_valid_layer_name(self) -> None:
        """
        Tests whether the LayerFactory correctly loads 
        a class if we provide it with a valid layer name.
        """
        layer = LayerFactory.get_layer_class('sparsey_layer')

        print(layer)

        assert issubclass(layer, torch.nn.Module)


    def test_invalid_layer_name(self) -> None:
        """
        Tests whether the LayerFactory throws an error  
        if we provide it with a invalid layer name.
        """
        with pytest.raises(ValueError):
            LayerFactory.get_layer_class('not_valid_layer')


    def test_sparsey_layer(self) -> None:
        """
        Tests whether the LayerFactory correctly constructs a Sparsey layer
        or not.
        """
        layer_obj = LayerFactory.create_layer(
            'sparsey_layer',
            num_macs=10, num_cms_per_mac=8,
            num_neurons_per_cm=4, mac_grid_num_rows=4,
            mac_grid_num_cols=4, mac_receptive_field_radius=0.5,
            prev_layer_cms_per_mac=10, prev_layer_neurons_per_cm=10,
            prev_layer_mac_positions=[
                (0.0, 0.0), (0.0, 0.5), (0.0, 1.0),
                (0.5, 0.0), (0.5, 0.5), (0.5, 1.0),
                (1.0, 0.0), (1.0, 0.5), (1.0, 1.0),
            ]
        )

        data = torch.randint(
            0, 2, (4, 9, 10, 10), dtype=torch.float32
        )

        assert layer_obj(data).shape == (4, 10, 8, 4)
