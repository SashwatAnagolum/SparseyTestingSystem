# -*- coding: utf-8 -*-

"""
Test Sparsey Model Configs: tests covering the config files for 
    customizing the structure of Sparsey models.
"""


import os
import pytest

from schema import SchemaError

from sparseypy.cli.config_validation.validate_config import (
    validate_config, get_config_info
)


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
        config_filepath = os.path.join(
            './test/unit/cli/config_validation/model',
            'valid_sparsey_config.yaml'
        )

        valid_schema = get_config_info(config_filepath)

        return valid_schema


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

        Test case ID: TC-01-01
        """
        validated_config = validate_config(
            sparsey_model_schema, 'model', 'sparsey'
        )

        assert isinstance(validated_config, dict)


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

        Test case ID: TC-01-16
        """
        del sparsey_model_schema['input_dimensions']

        with pytest.raises(SchemaError):
             validate_config(
                sparsey_model_schema, 'model', 'sparsey',
                survive_with_exception=True
            )


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

        Test case ID: TC-01-14
        """
        del sparsey_model_schema['layers']

        with pytest.raises(SchemaError):
            validate_config(
                sparsey_model_schema, 'model', 'sparsey',
                survive_with_exception=True
            )


    def test_out_of_bounds_activation_thresholds(
            self, sparsey_model_schema: dict) -> None:
        """
        Test whether the config validation throws an error
        or not when the activation thresholds specfied are out of
        bounds.
        
        Args:
            sparsey_model_schema: a dict containing the valid
            sparsey model schema to be used for testing, passed in 
            via pytest's fixture functionality.

        Test case ID: TC-01-02
        """
        sparsey_model_schema['layers'][0]['params'][
            'activation_threshold_min'
        ] = -0.1

        with pytest.raises(SchemaError):
            validate_config(
                sparsey_model_schema, 'model', 'sparsey',
                survive_with_exception=True
            )


    def test_permanence_high_boundary(
            self, sparsey_model_schema: dict) -> None:
        """
        Test whether the config validation accepts the
        high boundary of the permanence parameter as valid or not.
        
        Args:
            sparsey_model_schema: a dict containing the valid
            sparsey model schema to be used for testing, passed in 
            via pytest's fixture functionality.

        Test case ID: TC-01-12
        """
        sparsey_model_schema['layers'][0]['params'][
            'permanence_convexity'
        ] = 0.5

        validated_config = validate_config(
            sparsey_model_schema, 'model', 'sparsey'
        )

        assert isinstance(validated_config, dict)


    def test_permanence_low_boundary(
            self, sparsey_model_schema: dict) -> None:
        """
        Test whether the config validation throws an
        error when the permanence valus <= 0 or not.
        
        Args:
            sparsey_model_schema: a dict containing the valid
            sparsey model schema to be used for testing, passed in 
            via pytest's fixture functionality.

        Test case ID: TC-01-15
        """
        sparsey_model_schema['layers'][0]['params'][
            'permanence'
        ] = 0.0

        with pytest.raises(SchemaError):
            validate_config(
                sparsey_model_schema, 'model', 'sparsey',
                survive_with_exception=True
            )


    def test_receptive_field_radius_low_boundary(
            self, sparsey_model_schema: dict) -> None:
        """
        Test whether the config validation throws an error 
        when the receptive field size <= 0 or not.
        
        Args:
            sparsey_model_schema: a dict containing the valid
            sparsey model schema to be used for testing, passed in 
            via pytest's fixture functionality.

        Test case ID: TC-01-13
        """
        sparsey_model_schema['layers'][0]['params'][
            'mac_receptive_field_size'
        ] = 0.0

        with pytest.raises(SchemaError):
            validate_config(
                sparsey_model_schema, 'model', 'sparsey',
                survive_with_exception=True
            )
