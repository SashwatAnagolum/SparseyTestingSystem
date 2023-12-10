# -*- coding: utf-8 -*-

"""
Test MAC Layer: Tests to ensure that the MAC layer freezes weights correctly
based on the activity of the inputs.
"""

import pytest
import torch
from sparsepy.core.model_layers.sparsey_layer import MAC
from sparsepy.core.model_layers.sparsey_layer import SparseyLayer

@pytest.fixture
def layer(num_macs=10, num_cms_per_mac=5, num_neurons_per_mac=100, 
          mac_grid_num_rows=2, mac_grid_num_cols=5, mac_receptive_field_radius=1.0,
          prev_layer_num_macs=10, prev_layer_cms_per_mac=5, prev_layer_neurons_per_cm=20,
          prev_layer_mac_grid_num_rows=2, prev_layer_mac_grid_num_cols=5) -> SparseyLayer:
    """
    Fixture to create a Layer instance with a list of MACs for testing.
    """
    return SparseyLayer(num_macs, num_cms_per_mac, num_neurons_per_mac, 
                 mac_grid_num_rows, mac_grid_num_cols, mac_receptive_field_radius,
                 prev_layer_num_macs, prev_layer_cms_per_mac, prev_layer_neurons_per_cm,
                 prev_layer_mac_grid_num_rows, prev_layer_mac_grid_num_cols)

def test_mac_weight_freezing(layer):
    """
    Test that MAC weights within the layer are frozen based on input activity.
    """
    # Simulate input tensor with random data
    x = torch.randn((1, layer.num_macs * layer.num_neurons_per_mac))

    # Forward pass through the layer
    output = layer.forward(x)

    freezing_threshold = 0.5  # This threshold should be part of your freezing logic
    active_weights = (x > freezing_threshold).float().mean()
    
    for mac in layer.macs:
        if mac.should_freeze(active_weights):
            mac.freeze_weights()
 
    for mac in layer.macs:
        if active_weights > freezing_threshold:
            assert mac.are_weights_frozen(), f"Weights of MAC {mac} should be frozen."
        else:
            assert not mac.are_weights_frozen(), f"Weights of MAC {mac} should not be frozen."

