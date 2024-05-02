# -*- coding: utf-8 -*-

"""
Test Sparsey Trainer Configs: tests covering the config files for 
    customizing the structure of trainers for Sparsey models.
"""


import os
import pytest

from schema import SchemaError

from sparseypy.cli.config_validation.validate_config import (
    validate_config, get_config_info
)


class TestSparseyTrainerConfigs:
    """
    TestSparseyTrainerConfigs: class containing a collection
    of tests related to config files for Sparsey trainer creation.
    """
    @pytest.fixture
    def sparsey_trainer_schema(self) -> dict:
        """
        Returns a valid Sparsey trainer schema.

        Returns:
            a dict containing a valid Sparsey trainer schema
        """
        config_filepath = os.path.join(
            './test/unit/cli/config_validation/trainer',
            'valid_sparsey_trainer_config.yaml'
        )

        valid_schema = get_config_info(config_filepath)

        return valid_schema


    def test_valid_trainer_schema(
            self, sparsey_trainer_schema: dict) -> None:
        """
        Tests the config file validation for the case
        where the config file for a trainer is fully
        valid.

        Args:
            sparsey_trainer_schema: a dict containing the valid
            sparsey trainer schema to be used for testing, passed in 
            via pytest's fixture functionality.

        Test case ID: TC-01-03
        """
        validated_config = validate_config(
            sparsey_trainer_schema, 'training_recipe', 'sparsey',
            survive_with_exception=True
        )

        assert isinstance(validated_config, dict)


    def test_missing_optimizer_name(self, sparsey_trainer_schema: dict) -> None:
        """
        Tests the config file validation for the case
        where the config file for a Sparsey model is fully
        valid.

        Args:
            sparsey_trainer_schema: a dict containing the valid
            sparsey trainer schema to be used for testing, passed in 
            via pytest's fixture functionality.

        Test case ID: TC-01-17
        """
        del sparsey_trainer_schema['optimizer']['name']

        with pytest.raises(SchemaError):
            validate_config(
                sparsey_trainer_schema, 'training_recipe', 'sparsey',
                survive_with_exception=True
            )


    def test_invalid_optimizer_name(self, sparsey_trainer_schema: dict) -> None:
        """
        Tests the config file validation for the case
        where the config file for a Sparsey model is fully
        valid.

        Args:
            sparsey_trainer_schema: a dict containing the valid
            sparsey trainer schema to be used for testing, passed in 
            via pytest's fixture functionality.

        Test case ID: TC-01-18
        """
        sparsey_trainer_schema['optimizer']['name'] = 'invalid_name'

        with pytest.raises(SchemaError):
            validate_config(
                sparsey_trainer_schema, 'training_recipe', 'sparsey',
                survive_with_exception=True
            )


    def test_batch_size_lower_boundary_valid(
            self, sparsey_trainer_schema: dict) -> None:
        """
        TODO

        Args:
            sparsey_trainer_schema: a dict containing the valid
            sparsey trainer schema to be used for testing, passed in 
            via pytest's fixture functionality.

        Test case ID: TC-01-19
        """
        sparsey_trainer_schema['training']['dataloader']['batch_size'] = 1

        validated_config = validate_config(
            sparsey_trainer_schema, 'training_recipe', 'sparsey',
            survive_with_exception=True
        )

        assert isinstance(validated_config, dict)


    def test_batch_size_lower_boundary_invalid(
            self, sparsey_trainer_schema: dict) -> None:
        """
        TODO

        Args:
            sparsey_trainer_schema: a dict containing the valid
            sparsey trainer schema to be used for testing, passed in 
            via pytest's fixture functionality.

        Test case ID: TC-01-20
        """
        sparsey_trainer_schema['training']['dataloader']['batch_size'] = 0

        with pytest.raises(SchemaError):
            validate_config(
                sparsey_trainer_schema, 'training_recipe', 'sparsey',
                survive_with_exception=True
            )


    def test_num_epochs_lower_boundary_valid(
            self, sparsey_trainer_schema: dict) -> None:
        """
        TODO

        Args:
            sparsey_trainer_schema: a dict containing the valid
            sparsey trainer schema to be used for testing, passed in 
            via pytest's fixture functionality.

        Test case ID: TC-01-21
        """
        sparsey_trainer_schema['training']['num_epochs'] = 1

        validated_config = validate_config(
            sparsey_trainer_schema, 'training_recipe', 'sparsey',
            survive_with_exception=True
        )

        assert isinstance(validated_config, dict)


    def test_num_epochs_lower_boundary_invalid(
            self, sparsey_trainer_schema: dict) -> None:
        """
        TODO

        Args:
            sparsey_trainer_schema: a dict containing the valid
            sparsey trainer schema to be used for testing, passed in 
            via pytest's fixture functionality.

        Test case ID: TC-01-22
        """
        sparsey_trainer_schema['training']['num_epochs'] = 0

        with pytest.raises(SchemaError):
            validate_config(
                sparsey_trainer_schema, 'training_recipe', 'sparsey',
                survive_with_exception=True
            )


    def test_missing_metric_list(self, sparsey_trainer_schema: dict) -> None:
        """
        TODO

        Args:
            sparsey_trainer_schema: a dict containing the valid
            sparsey trainer schema to be used for testing, passed in 
            via pytest's fixture functionality.

        Test case ID: TC-01-04
        """
        del sparsey_trainer_schema['metrics']

        with pytest.raises(SchemaError):
            validate_config(
                sparsey_trainer_schema, 'training_recipe', 'sparsey',
                survive_with_exception=True
            )


    def test_no_listed_metrics(self, sparsey_trainer_schema: dict) -> None:
        """
        Tests the config file validation for the case
        where the config file for a Sparsey model is fully
        valid.

        Args:
            sparsey_trainer_schema: a dict containing the valid
            sparsey trainer schema to be used for testing, passed in 
            via pytest's fixture functionality.

        Test case ID: TC-01-23
        """
        sparsey_trainer_schema['metrics'] = []

        with pytest.raises(SchemaError):
            validate_config(
                sparsey_trainer_schema, 'training_recipe', 'sparsey',
                survive_with_exception=True
            )
