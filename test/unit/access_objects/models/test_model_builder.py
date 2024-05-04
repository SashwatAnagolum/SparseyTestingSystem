# -*- coding: utf-8 -*-

import pytest
import torch

from sparseypy.access_objects.models.model_builder import ModelBuilder
from sparseypy.core.model_layers.sparsey_layer import SparseyLayer

class TestModelBuilder:
    """
    Class to test the functionality of the ModelBuilder class.
    """

    def test_sparsey_layer_creation_valid(self):
        """
        Tests whether the ModelBuilder correctly constructs a model with a valid Sparsey layer.

        Test case ID: TC-09-01
        """
        model_config = {
            'layers': [
                {
                    'name': 'sparsey',
                    'params': {
                        'autosize_grid': True, 
                        'grid_layout': 'hex',
                        'num_macs': 25, 
                        'num_cms_per_mac': 5, 
                        'num_neurons_per_cm': 5,
                        'mac_grid_num_rows': 5, 
                        'mac_grid_num_cols': 5,
                        'mac_receptive_field_size': 1.5, 
                        'prev_layer_num_cms_per_mac': 12,
                        'prev_layer_num_neurons_per_cm': 10,
                        'prev_layer_mac_grid_num_rows': 4,
                        'prev_layer_mac_grid_num_cols': 6,
                        'prev_layer_num_macs': 24, 
                        'prev_layer_grid_layout': 'rect',
                        'layer_index': 6, 
                        'sigmoid_phi': 0.1, 
                        'sigmoid_lambda': 0.9,
                        'saturation_threshold': 0.8,
                        'permanence_steps': 0.1, 
                        'permanence_convexity': 0.1,
                        'activation_threshold_min': 0.4, 
                        'activation_threshold_max': 0.8,
                        'min_familiarity': 0.5, 
                        'sigmoid_chi': 1.2
                    }
                }
            ]
        }

        device = torch.device("cpu")
        model = ModelBuilder.build_model(model_config, device)
        layer_obj = model.get_submodule('Layer_0')

        assert isinstance(layer_obj, SparseyLayer), "Model does not contain a valid SparseyLayer"

    def test_sparsey_layer_creation_invalid(self):
        """
        Tests whether the ModelBuilder throws an error when invalid parameters are passed during the creation of a SparseyLayer.

        Test case ID: TC-09-02
        """
        model_config = {
            'layers': [
                {
                    'name': 'sparsey',
                    'params': {
                        'autosize_grid': True, 
                        'grid_layout': 'hex',
                        'num_macs': 25, 
                        'num_cms_per_mac': 5, 
                        'num_neurons_per_cm': 5,
                        'mac_grid_num_rows': 5, 
                        'mac_grid_num_cols': 5,
                        'mac_receptive_field_size': 1.5, 
                        'prev_layer_num_cms_per_mac': 12,
                        'prev_layer_num_neurons_per_cm': 10,
                        'prev_layer_mac_grid_num_rows': 4,
                        'prev_layer_mac_grid_num_cols': 6,
                        'prev_layer_num_macs': 24, 
                        'prev_layer_grid_layout': 'rect',
                        'layer_index': 6, 
                        'sigmoid_phi': 0.1, 
                        'sigmoid_lambda': 0.9,
                        'saturation_threshold': 0.8,
                        'permanence_steps': 0.1, 
                        'permanence_convexity': 0.1,
                        'activation_threshold_min': 0.4, 
                        'activation_threshold_max': 0.8,
                        'min_familiarity': 0.5,
                        'invalid_param': 0.6  # Intentionally invalid to test error handling
                    }
                }
            ]
        }

        device = torch.device("cpu")
        with pytest.raises(Exception):
            ModelBuilder.build_model(model_config, device)

