# -*- coding: utf-8 -*-

"""
Test Sparsey Layer: tests covering the functionality of a single
    MAC and a Sparsey layer.
"""


from typing import Tuple

import torch
import pytest

from sparseypy.core.model_layers.sparsey_layer import MAC, SparseyLayer


class TestMAC:
    @pytest.fixture
    def sample_sparsey_layer(self):
        """
        Returns a sample SparseyLayer object to perform
        tests with.
        """
        sparsey_layer = SparseyLayer(
            autosize_grid=False,
            grid_layout="rect",
            num_macs=16,
            num_cms_per_mac=8,
            num_neurons_per_cm=16,
            mac_grid_num_rows=4,
            mac_grid_num_cols=4,
            prev_layer_num_macs=9,
            mac_receptive_field_size=0.5,
            prev_layer_num_cms_per_mac=12,
            prev_layer_num_neurons_per_cm=10,
            prev_layer_mac_grid_num_rows=3,
            prev_layer_mac_grid_num_cols=3,
            prev_layer_grid_layout="rect",
            layer_index=2,
            sigmoid_phi=5.0,
            sigmoid_lambda=28.0,
            saturation_threshold=0.5,
            permanence_steps=1.0,
            permanence_convexity=1.0,
            activation_threshold_max=1.0,
            activation_threshold_min=0.2,
            min_familiarity=0.2,
            sigmoid_chi=2.5,
            device=torch.device("cpu")           
        )

        return sparsey_layer
    
    @pytest.fixture
    def mac_config(self):
        return {
        'num_cms': 2,
        'num_neurons': 2,
        'input_filter': torch.tensor([0, 1, 2, 3]),  # Mock input filter
        'num_cms_per_mac_in_input': 2,
        'num_neurons_per_cm_in_input': 2,
        'layer_index': 0,
        'mac_index': 0,
        'sigmoid_lambda': 28.0,
        'sigmoid_phi': 5.0,
        'activation_threshold_min': 0.2,
        'activation_threshold_max': 1,
        'sigmoid_chi': 2.5,
        'min_familiarity': 0.2,
        'permanence_steps': 1.0,
        'permanence_convexity': 1.0,
        'device': torch.device('cpu'),
        'prev_layer_num_cms_per_mac': 2,
        'prev_layer_num_neurons_per_cm': 2,
        }

    @pytest.fixture
    def mock_input(self):
        return torch.rand(1, 4, 2, 2)  # should find one that is actually a real input when running


    def test_mac_output_shape(self, mac_config, mock_input):
        """
        Test that the output shape of the mac is correct
        TC-04-01: The shape of the output tensor for a MAC is correct, depending on the parameters of the MAC and the size of the input data
        """
        mac = MAC(**mac_config)
        # Pass the mock input through the MAC
        output = mac(mock_input)
        expected_shape = (1, mac_config['num_cms'], mac_config['num_neurons'])
        # Verify the output shape
        assert output.shape == expected_shape, f"Expected output shape {expected_shape}, but got {output.shape}"



    def test_output_sparsity(self, sample_sparsey_layer):
        """
        Test that each CM in the output of a Sparsey Layer contains only
        one active neuron.
        TC-04-02: Test for correct operation of the winner-take-all modules and sparsity of the Sparsey layer output
        """
        data = torch.randint(0, 2, (32, 9, 120))
        output = sample_sparsey_layer(data)

        equal_elements_one = torch.eq(
            output,
            torch.ones(output.shape, dtype=torch.float32)
        )

        equal_elements_zero = torch.eq(
            output,
            torch.zeros(output.shape, dtype=torch.float32)
        )

        assert (
            torch.sum(equal_elements_one).item() +
            torch.sum(equal_elements_zero)
         ) == 32 * 16 * 8 * 16
