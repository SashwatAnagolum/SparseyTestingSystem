# -*- coding: utf-8 -*-

"""
Test Default HPO Configs: tests covering the config files for 
    customizing HPO runss.
"""


import os
import pytest

from schema import SchemaError

from sparseypy.cli.config_validation.validate_config import (
    validate_config, get_config_info
)


class TestDefaultHPOConfigs:
    """
    TestDefaultHPOConfigs: class containing a collection
    of tests related to config files for HPO run creation.
    """
    @pytest.fixture
    def default_hpo_schema(self) -> dict:
        """
        Returns a valid HPO schema.

        Returns:
            a valid HPO schema
        """
        config_filepath = os.path.join(
            './test/unit/cli/config_validation/hpo',
            'valid_hpo_config.yaml'
        )

        valid_schema = get_config_info(config_filepath)

        return valid_schema
    

    def test_valid_hpo_schema(
            self, default_hpo_schema: dict) -> None:
        """
        Tests the config file validation for the case
        where the config file for a HPO run is fully
        valid.

        Test case ID: TC-01-08

        Args:
            default_hpo_schema: a dict containing the valid
                sparsey model schema to be used for testing,
                passed in via pytest's fixture functionality.
        """
        validated_config = validate_config(
            default_hpo_schema, 'hpo', 'default'
        )

        assert isinstance(validated_config, dict)


    def test_incorrect_strategy_name(
            self, default_hpo_schema: dict) -> None:
        """
        Tests the config file validation for the case
        where the config file for a HPO run contains an incorrect
        HPO strategy.

        Test case ID: TC-01-09

        Args:
            default_hpo_schema: a dict containing the valid
                sparsey model schema to be used for testing,
                passed in via pytest's fixture functionality.
        """
        default_hpo_schema['hpo_strategy'] = 'invalid_strategy'

        with pytest.raises(SchemaError):
            validate_config(
                default_hpo_schema, 'hpo', 'default',
                survive_with_exception=True
            )


    def test_uncomputed_metrics(
            self, default_hpo_schema: dict) -> None:
        """
        Tests the config file validation for the case
        where the config file for a HPO run contains uncomputed
        metrics in the objective.

        Test case ID: TC-01-10

        Args:
            default_hpo_schema: a dict containing the valid
                sparsey model schema to be used for testing,
                passed in via pytest's fixture functionality.
        """
        default_hpo_schema['metrics'][0]['name'] = 'basis_average'

        with pytest.raises(SchemaError):
            validate_config(
                default_hpo_schema, 'hpo', 'default',
                survive_with_exception=True
            )


    def test_missing_metric_list(
            self, default_hpo_schema: dict) -> None:
        """
        Tests the config file validation for the case
        where the config file for a HPO run does not contain
        a list of metrics to compute.

        Test case ID: TC-01-11

        Args:
            default_hpo_schema: a dict containing the valid
                sparsey model schema to be used for testing,
                passed in via pytest's fixture functionality.
        """
        del default_hpo_schema['metrics']

        with pytest.raises(SchemaError):
            validate_config(
                default_hpo_schema, 'hpo', 'default',
                survive_with_exception=True
            )


    def test_incorrect_hyperparam_specification(
            self, default_hpo_schema: dict) -> None:
        """
        Tests the config file validation for the case
        where the config file for a HPO run contains
        incorrectly specified hyperparameters.

        Test case ID: TC-01-24

        Args:
            default_hpo_schema: a dict containing the valid
                sparsey model schema to be used for testing,
                passed in via pytest's fixture functionality.
        """
        default_hpo_schema['hyperparameters']['num_layers']['value'] = [3, 4]

        with pytest.raises(SchemaError):
            validate_config(
                default_hpo_schema, 'hpo', 'default',
                survive_with_exception=True
            )
