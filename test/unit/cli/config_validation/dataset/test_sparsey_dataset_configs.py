# -*- coding: utf-8 -*-

"""
Test Sparsey Dataset Configs: tests covering the config files for 
    customizing the structure of Sparsey datasets.
"""

import pytest
import os
from sparsepy.cli.config_validation.validate_config import validate_config
import copy
from schema import SchemaError, SchemaMissingKeyError


class TestSparseyDatasetConfigs:
    """
    TestSparseyDatasetConfigs: class containing a collection
    of tests related to config files for Sparsey dataset creation.
    """
    @pytest.fixture
    def sparsey_dataset_schema(self) -> dict:
        """
        Returns a valid Sparsey dataset schema.

        Returns:
            a valid Sparsey dataset schema
        """
        # Ensure the directory path is valid for the test environment
        valid_data_dir = ".\demo\sample_mnist_dataset"
        valid_schema = {
            'dataset_type': 'image',
            'params': {
                'data_dir': valid_data_dir,
                'image_format': '.png'
            },
            'preprocessed': True,
            'preprocessed_stack': {
                'transform_list': [
                    {
                        'name': 'resize',
                        'params': {
                            'size': [8, 8],
                            'antialias': True
                        }
                    },
                    {
                        'name': 'binarize_transform',
                        'params': {
                            'binarize_threshold': 0.5
                        }
                    },
                    {
                        'name': 'skeletonization_transform',
                        'params': {
                            'sigma': 3
                        }
                    },
                ]
            }
        }

        return valid_schema

    def perform_assertion(self, schema: dict, expected_value: bool) -> None:
        """
        Performs an assertion based on the return value
        of a call to validate_config.
        """
        assert validate_config(
            schema, 'dataset',
            'image'
        )[1] == expected_value

    def test_valid_sparsey_dataset_schema(self, sparsey_dataset_schema: dict) -> None:
        """
        Tests the config file validation for the case
        where the config file for a Sparsey dataset is fully
        valid.
        """
        self.perform_assertion(sparsey_dataset_schema, True)

    def test_missing_data_dir(self, sparsey_dataset_schema: dict) -> None:
        """
        Tests the config file validation for the case
        where the data directory is missing.
        """
        del sparsey_dataset_schema['params']['data_dir']
        with pytest.raises(SchemaError):
            validate_config(sparsey_dataset_schema, 'dataset', 'image')

    def test_invalid_image_format(self, sparsey_dataset_schema: dict) -> None:
        """
        Tests the config file validation for the case
        where the image format is invalid.
        """
        sparsey_dataset_schema['params']['image_format'] = 'jpg'  # Missing leading dot
        with pytest.raises(SchemaError):
            validate_config(sparsey_dataset_schema, 'dataset', 'image')

    def test_preprocessed_without_stack(self, sparsey_dataset_schema: dict) -> None:
        """
        Tests the config file validation for the case
        where preprocessed is True but preprocessed_stack is missing.
        """
        del sparsey_dataset_schema['preprocessed_stack']
        with pytest.raises(SchemaMissingKeyError):
            validate_config(sparsey_dataset_schema, 'dataset', 'image')

    def test_invalid_preprocessed_stack(self, sparsey_dataset_schema: dict) -> None:
        """
        Tests the config file validation for the case
        where the preprocessed_stack is invalid.
        """
        sparsey_dataset_schema['preprocessed_stack']['transform_list'][0]['name'] = "anything"
        self.perform_assertion(sparsey_dataset_schema, False)