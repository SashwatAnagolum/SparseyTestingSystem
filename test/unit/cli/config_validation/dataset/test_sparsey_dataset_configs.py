# -*- coding: utf-8 -*-

"""
Test Sparsey Dataset Configs: tests covering the config files for 
    customizing the structure of Sparsey datasets.
"""


import os
import copy

import pytest
from schema import SchemaError, SchemaMissingKeyError

from sparseypy.cli.config_validation.validate_config import (
    validate_config, get_config_info
)


class TestSparseyDatasetConfigs:
    """
    TestSparseyDatasetConfigs: class containing a collection
    of tests related to config files for Sparsey dataset creation.
    """
    @pytest.fixture
    def sparsey_dataset_config(self) -> dict:
        """
        Returns a valid image dataset config.

        Returns:
            (dict): a vliad dataset config.
        """
        config_filepath = os.path.join(
            './test/unit/cli/config_validation/dataset',
            'valid_sparsey_dataset_config.yaml'
        )

        valid_schema = get_config_info(config_filepath)

        return valid_schema


    def test_valid_sparsey_dataset_schema(
            self, sparsey_dataset_config: dict) -> None:
        """
        Tests the config file validation for the case
        where the config file for a Sparsey dataset is fully
        valid.

        Test case ID: TC-01-05

        Args:
            sparsey_dataset_config (dict): a fully valid
                dataset config.
        """
        validated_config = validate_config(
            sparsey_dataset_config, 'dataset',
            'image'
        )

        assert isinstance(validated_config, dict)


    def test_missing_data_dir(self, sparsey_dataset_config: dict) -> None:
        """
        Tests the config file validation for the case
        where the data directory is missing.

        Test case ID: TC-01-06

        Args:
            sparsey_dataset_config (dict): a fully valid
                dataset config.
        """
        del sparsey_dataset_config['params']['data_dir']

        with pytest.raises(SchemaError):
            validate_config(
                sparsey_dataset_config, 'dataset', 'image',
                survive_with_exception=True
                )


    def test_missing_preprocessing_stack(
            self, sparsey_dataset_config: dict) -> None:
        """
        Tests the config file validation for the case
        where preprocessed is True but preprocessed_stack is missing.

        Test case ID: TC-01-07

        Args:
            sparsey_dataset_config (dict): a fully valid
                dataset config.
        """
        del sparsey_dataset_config['preprocessed_stack']
        # made optional
        # with pytest.raises(SchemaError):
        assert validate_config(
                sparsey_dataset_config, 'dataset', 'image',
                survive_with_exception=True
                ) is not None


    def test_invalid_preprocessed_stack(
            self, sparsey_dataset_config: dict) -> None:
        """
        Tests the config file validation for the case
        where the preprocessed_stack is invalid.

        Test case ID: TC-01-25

        Args:
            sparsey_dataset_config (dict): a fully valid
                dataset config.
        """
        sparsey_dataset_config[
            'preprocessed_stack'
        ]['transform_list'][0]['name'] = 'invalid_transform'

        with pytest.raises(SchemaError):
            validate_config(
                sparsey_dataset_config, 'dataset', 'image',
                survive_with_exception=True
            )
