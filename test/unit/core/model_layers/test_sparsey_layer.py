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
