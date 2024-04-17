import pytest
import torch
from sparseypy.core.model_layers.sparsey_layer import MAC 

@pytest.fixture
def mac_config():
    return {
        'num_cms': 2,
        'num_neurons': 2,
        'input_filter': torch.tensor([0, 1, 2, 3]),  # Mock input filter, adjust as needed
        'num_cms_per_mac_in_input': 2,
        'num_neurons_per_cm_in_input': 2,
        'layer_index': 0,
        'mac_index': 0,
        'sigmoid_lambda': 28.0,
        'sigmoid_phi': 5.0,
        'permanence_steps' : 0.1, 
        'permanence_convexity' : 0.1,
        'activation_threshold_min': 0.2,
        'activation_threshold_max': 1,
        'sigmoid_chi': 2.5,
        'min_familiarity': 0.2,
    }

@pytest.fixture
def mock_input():
    return torch.rand(1, 4, 2, 2)  # should find one that is actually a real input when running

def test_mac_output_shape(mac_config, mock_input):
    mac = MAC(**mac_config)
    # Pass the mock input through the MAC
    output = mac(mock_input)
    expected_shape = (1, mac_config['num_cms'], mac_config['num_neurons'])
    # Verify the output shape
    assert output.shape == expected_shape, f"Expected output shape {expected_shape}, but got {output.shape}"
