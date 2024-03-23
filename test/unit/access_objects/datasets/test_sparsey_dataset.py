# -*- coding: utf-8 -*-

"""
Test Sparsey Dataset: test cases for the SparseyDataset class.
"""

import pytest
import torch

from sparseypy.access_objects.datasets import SparseyDataset

class TestSparseyDataset:
    """
    TestSparseyDataset: a class holding a collection
        of tests focused on the SparseyDataset class.
    """
    def test_valid_dataset_size(self) -> None:
        """
        Tests whether the SparseyDataset instantiates with 
        the correct size if given a 16x24 dataset.
        """
        ds = SparseyDataset('./datasets/MNIST', width=16, height=24)

        # the example MNIST dataset extract contains 1000 items
        assert len(ds) == 1000

    def test_valid_dataset_dimensions(self) -> None:
        """
        Tests whether the SparseyDataset applies the width and height
        in the correct order given a 16x24 dataset.
        """
        ds = SparseyDataset('./datasets/MNIST', width=16, height=24)

        next_item = next(iter(ds))[0]

        assert next_item.size() == (1, 24, 16)

    def test_single_item_retrieval(self) -> None:
        """
        Tests whether the SparseyDataset retrieves the correct
        input item for a one-item dataset.
        """
        # retrieve the 8x5 example item from its one-item dataset
        ds = SparseyDataset('./test/unit/access_objects/datasets/sparsey_dataset/one_item',
                            width=8, height=5)

        next_input = next(iter(ds))[0]

        # BUG ensure the binary input is read in the correct order
        # validate non-square input
        expected_bytes = torch.Tensor([[
            [255, 0, 0, 0, 0, 0, 255, 0],
            [0, 0, 0, 0, 255, 0, 0, 0],
            [0, 255, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 255, 0, 255, 255, 255, 255]
            ]])

        # old rotated version
        # expected_bytes = torch.Tensor([[
        #     [255, 0, 0, 0, 0],
        #     [0, 0, 255, 0, 0],
        #     [0, 0, 0, 0, 255],
        #     [0, 0, 0, 0, 0],
        #     [0, 255, 0, 0, 255],
        #     [0, 0, 0, 0, 255],
        #     [255, 0, 0, 0, 255],
        #     [0, 0, 0, 0, 255]
        # ]])

        assert torch.equal(next_input, expected_bytes)
