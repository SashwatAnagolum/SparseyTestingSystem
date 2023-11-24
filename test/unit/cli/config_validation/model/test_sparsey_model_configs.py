# -*- coding: utf-8 -*-

"""
Test Sparsey Model Configs: tests covering the config files for 
    customizing the structure of Sparsey models.
"""


import pytest

from sparsepy.cli.config_validation.validate_config import validate_config


class TestSparseyModelConfigs:
    """
    TestSparseyModelConfigs: class containing a collection
    of tests related to config files for Sparsey model creation.
    """
    @pytest.fixture
    def sparsey_model_schema(self) -> dict:
        """
        Returns a valid Sparsey model schema.

        Returns:
            a valid Sparsey model schema
        """
        valid_schema = {
            'input_dimensions': {
                'width': 28,
                'height': 28
            }, 
            'num_layers': 5,
            'layerwise_configs': {
                'num_macs': [50, 40, 30, 20, 10],
                'mac_grid_num_rows': [5, 4, 3, 2, 1],
                'mac_grid_num_cols': [10, 10, 10, 10, 10],
                'num_cms_per_mac': [5, 5, 5, 5, 5],
                'num_neurons_per_cm': [10, 10, 10, 10, 10],
                'receptive_field_radii': [2.0, 2.0, 5.1, 3.2, 4.2]
            }
        }

        return valid_schema


    def perform_assertion(self, schema: dict, expected_value: bool) -> None:
        """
        Performs an assertion based on the return value
        of a call to validate_config.
        """
        assert validate_config(
            schema, 'model',
            'sparsey'
        )[1] == expected_value


    def test_valid_sparsey_model_schema(
            self, sparsey_model_schema: dict) -> None:
        """
        Tests the config file validation for the case
        where the config file for a Sparsey model is fully
        valid.

        Args:
            sparsey_model_schema: a dict containing the valid
            sparsey model schema to be used for testing, passed in 
            via pytest's fixture functionality.
        """
        self.perform_assertion(sparsey_model_schema, True)


    def test_missing_num_layers(
            self, sparsey_model_schema: dict) -> None:
        """
        Tests the config file validation for the case
        where the config file for a Sparsey model is fully
        valid.

        Args:
            sparsey_model_schema: a dict containing the valid
            sparsey model schema to be used for testing, passed in 
            via pytest's fixture functionality.
        """
        del sparsey_model_schema['num_layers']

        self.perform_assertion(sparsey_model_schema, False)


    def test_missing_input_dimensions(
            self, sparsey_model_schema: dict) -> None:
        """
        Tests the config file validation for the case
        where the config file for a Sparsey model is fully
        valid.

        Args:
            sparsey_model_schema: a dict containing the valid
            sparsey model schema to be used for testing, passed in 
            via pytest's fixture functionality.
        """
        del sparsey_model_schema['input_dimensions']

        self.perform_assertion(sparsey_model_schema, False)


    def test_missing_layerwise_configs(
            self, sparsey_model_schema: dict) -> None:
        """
        Tests the config file validation for the case
        where the config file for a Sparsey model is fully
        valid.

        Args:
            sparsey_model_schema: a dict containing the valid
            sparsey model schema to be used for testing, passed in 
            via pytest's fixture functionality.
        """
        del sparsey_model_schema['layerwise_configs']

        self.perform_assertion(sparsey_model_schema, False)


    def test_incorrect_data_type_layerwise_configs(
            self, sparsey_model_schema: dict) -> None:
        """
        Tests the config file validation for the case
        where the config file for a Sparsey model is fully
        valid.

        Args:
            sparsey_model_schema: a dict containing the valid
            sparsey model schema to be used for testing, passed in 
            via pytest's fixture functionality.
        """
        sparsey_model_schema['layerwise_configs']['num_macs'][0] = 50.0

        self.perform_assertion(sparsey_model_schema, False)


    def test_incorrect_length_layerwise_configs(
            self, sparsey_model_schema: dict) -> None:
        """
        Tests the config file validation for the case
        where the config file for a Sparsey model is fully
        valid.

        Args:
            sparsey_model_schema: a dict containing the valid
            sparsey model schema to be used for testing, passed in 
            via pytest's fixture functionality.
        """
        sparsey_model_schema['layerwise_configs']['num_cms_per_mac'].append(
            45
        )

        self.perform_assertion(sparsey_model_schema, False)


    def test_negative_num_layers(
            self, sparsey_model_schema: dict) -> None:
        """
        Tests the config file validation for the case
        where the config file for a Sparsey model is fully
        valid.

        Args:
            sparsey_model_schema: a dict containing the valid
            sparsey model schema to be used for testing, passed in 
            via pytest's fixture functionality.
        """
        sparsey_model_schema['num_layers'] = -2

        self.perform_assertion(sparsey_model_schema, False)


    def test_negative_layerwise_configs(
            self, sparsey_model_schema: dict) -> None:
        """
        Tests the config file validation for the case
        where the config file for a Sparsey model is fully
        valid.

        Args:
            sparsey_model_schema: a dict containing the valid
            sparsey model schema to be used for testing, passed in 
            via pytest's fixture functionality.
        """
        sparsey_model_schema['layerwise_configs']['mac_grid_num_cols']= -3

        self.perform_assertion(sparsey_model_schema, False)
