# -*- coding: utf-8 -*-

"""
Test Sparsey Layer: tests covering the functionality of a single
    MAC and a Sparsey layer.
"""


from typing import Tuple

import torch
import pytest

from sparsepy.core.model_layers.sparsey_layer import MAC, SparseyLayer


class TestMAC:
    @pytest.fixture
    def sample_sparsey_layer(self):
        """
        Returns a sample SparseyLayer object to perform
        tests with.
        """
        sparsey_layer = SparseyLayer(
            num_macs=12, num_cms_per_mac=8,
            num_neurons_per_cm=16, mac_grid_num_rows=4,
            mac_grid_num_cols=4, mac_receptive_field_radius=0.5,
            prev_layer_cms_per_mac=12, prev_layer_neurons_per_cm=10,
            prev_layer_mac_positions=[
                (0.0, 0.0), (0.0, 0.5), (0.0, 1.0),
                (0.5, 0.0), (0.5, 0.5), (0.5, 1.0),
                (1.0, 0.0), (1.0, 0.5), (1.0, 1.0),
            ]            
        )

        return sparsey_layer


    @pytest.mark.parametrize(
            'bsz, num_cms, num_neurons, input_filter,' + 
            ' prev_num_cms, prev_num_neurons, output_shape',
            [
                (16, 5, 5, [1, 2, 3], 10, 8, (16, 5, 5)),
                (8, 5, 8, [1, 2], 10, 8, (8, 5, 8)),
                (16, 3, 3, [1, 4], 2, 2, (16, 3, 3)),
                (1, 16, 8, [1], 4, 8, (1, 16, 8)),
                (4, 4, 4, [1, 2, 7], 10, 10, (4, 4, 4))
            ]
    )
    def test_mac_output_shape(
        self, bsz: int, num_cms: int, num_neurons: int,
        input_filter: list[int], prev_num_cms: int,
        prev_num_neurons: int, output_shape: Tuple[int, int, int]
    ):
        """
        Test the correctness of the output shape of the forward
        pass of a MAC.

        Args:

        """
        mac = MAC(
            num_cms, num_neurons, torch.Tensor(input_filter).long(),
            prev_num_cms, prev_num_neurons
        )

        data = torch.randint(
            0, 2, (bsz, 10, prev_num_cms, prev_num_neurons),
            dtype=torch.float32
        )

        output = mac(data)

        assert tuple(output.shape) == output_shape


    def test_mac_invalid_input_shape(self):
        """
        Test whether a MAC raises a ValueError when you pass
        an Tensor with an invalid data shape through it.
        """
        mac = MAC(
            10, 10, torch.Tensor([1, 2, 3]).long()  , 5, 5
        )

        data = torch.randint(0, 2, (32, 10, 4, 5))

        with pytest.raises(ValueError):
            mac(data)


    def test_sparsey_layer_valid_input_shape(self, sample_sparsey_layer):
        """
        Test whether a SparseyLayer outputs a Tensor of the right 
        shape when you pass a Tensor with a valid data shape through it.
        """
        data = torch.randint(0, 2, (32, 9, 12, 10))

        assert tuple(sample_sparsey_layer(data).shape) == (32, 12, 8, 16)


    def test_sparsey_layer_invalid_input_shape(self, sample_sparsey_layer):
        """
        Test whether a SparseyLayer raises a ValueError when you pass
        an Tensor with an invalid data shape through it.
        """
        data = torch.randint(0, 2, (32, 12, 11, 10))

        with pytest.raises(ValueError):
            sample_sparsey_layer(data)


    def test_output_sparsity(self, sample_sparsey_layer):
        """
        Test that each CM in the output of a Sparsey Layer contains only
        one active neuron.
        """
        data = torch.randint(0, 2, (32, 9, 12, 10))
        output = sample_sparsey_layer(data)

        assert tuple(output.shape) == (32, 12, 8, 16)

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
         ) == 32 * 12 * 8 * 16
