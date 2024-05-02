# -*- coding: utf-8 -*-

"""
Test Layer Factory: test cases for the LayerFactory class.
"""


import pytest
import torch

from sparseypy.core.model_layers.layer_factory import LayerFactory
from sparseypy.core.model_layers.sparsey_layer import SparseyLayer


class TestLayerFactory:
    """
    TestLayerFactory: a class holding a collection
        of tests focused on the LayerFactory class.
    """
    def test_valid_layer_name(self) -> None:
        """
        Tests whether the LayerFactory correctly loads 
        a class if we provide it with a valid layer name.

        Test case ID: TC-04-03
        """
        layer = LayerFactory.get_layer_class('sparsey')

        assert issubclass(layer, torch.nn.Module)


    def test_invalid_layer_name(self) -> None:
        """
        Tests whether the LayerFactory throws an error  
        if we provide it with a invalid layer name.

        Test case ID: TC-04-04
        """
        with pytest.raises(ValueError):
            LayerFactory.get_layer_class('not_valid_layer')


    def test_sparsey_layer_creation_valid(self) -> None:
        """
        Tests whether the LayerFactory correctly constructs a Sparsey layer
        or not.

        Test case ID: TC-04-05
        """
        layer_obj = LayerFactory.create_layer(
            'sparsey',
            autosize_grid=True, grid_layout='hex',
            num_macs=25, num_cms_per_mac=5, num_neurons_per_cm=5,
            mac_grid_num_rows=5, mac_grid_num_cols=5,
            mac_receptive_field_size=1.5, prev_layer_num_cms_per_mac=12,
            prev_layer_num_neurons_per_cm=10,
            prev_layer_mac_grid_num_rows=4,
            prev_layer_mac_grid_num_cols=6,
            prev_layer_num_macs=24, prev_layer_grid_layout='rect',
            layer_index=6, sigmoid_phi=0.1, sigmoid_lambda=0.9,
            saturation_threshold=0.8,
            permanence_steps=0.1, permanence_convexity=0.1, 
            activation_threshold_min=0.4, activation_threshold_max=0.8,
            min_familiarity=0.5, sigmoid_chi=1.2, device=torch.device("cpu")
        )

        assert isinstance(layer_obj, SparseyLayer)


    def test_sparsey_layer_creation_invalid(self) -> None:
        """
        Tests whether the LayerFactory throws an error when
        invalid parameters are passed during creation of a SparseyLayer.

        Test case ID: TC-04-06
        """
        with pytest.raises(TypeError):
            LayerFactory.create_layer(
                'sparsey',
                autosize_grid=True, grid_layout='hex',
                num_macs=25, num_cms_per_mac=5, num_neurons_per_cm=5,
                mac_grid_num_rows=5, mac_grid_num_cols=5,
                mac_receptive_field_size=1.5, prev_layer_num_cms_per_mac=12,
                prev_layer_num_neurons_per_cm=10,
                prev_layer_mac_grid_num_rows=4,
                prev_layer_mac_grid_num_cols=6,
                prev_layer_num_macs=24, prev_layer_grid_layout='rect',
                layer_index=6, sigmoid_phi=0.1, sigmoid_lambda=0.9,
                saturation_threshold=0.8,
                permanence=0.6, activation_threshold_min=0.4,
                activation_threshold_max=0.8,
                min_familiarity=0.5
            )
