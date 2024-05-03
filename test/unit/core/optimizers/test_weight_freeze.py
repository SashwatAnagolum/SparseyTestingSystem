import pytest
import torch
from sparseypy.core.model_layers.sparsey_layer import MAC, SparseyLayer

@pytest.fixture
def setup_sparsey_layer():
    """
    Fixture to create a SparseyLayer instance with essential constructor parameters.
    Additional attributes will be set directly on the layer object as needed.
    """
    # Parameters for the constructor
    layer = SparseyLayer(
        autosize_grid=True,
        grid_layout='rectangular',
        num_macs=10,
        num_cms_per_mac=5,
        num_neurons_per_cm=100,
        mac_grid_num_rows=2,
        mac_grid_num_cols=5,
        mac_receptive_field_size=1.0,
        prev_layer_num_cms_per_mac=5,
        prev_layer_num_neurons_per_cm=20,
        prev_layer_mac_grid_num_rows=2,
        prev_layer_mac_grid_num_cols=5,
        prev_layer_num_macs=10,
        prev_layer_grid_layout='rectangular',
        layer_index=1,
        sigmoid_phi=0.5,
        sigmoid_lambda=0.5,
        saturation_threshold=0.1,
        permanence_steps=0.1,
        permanence_convexity=0.1,
        activation_threshold_min=0.2,
        activation_threshold_max=0.8,
        min_familiarity=0.1,
        sigmoid_chi=0.1,
        device=torch.device('cpu')
    )

    return layer


def test_sparsey_layer_functionality(setup_sparsey_layer):
    """
    A test to verify some functionality of the SparseyLayer.
    """
    layer = setup_sparsey_layer
    inputs = torch.randn(1, 10, 100)
    output = layer(inputs)
    assert output.shape == torch.Size([1, 10, 500]), "Output dimensions are incorrect"