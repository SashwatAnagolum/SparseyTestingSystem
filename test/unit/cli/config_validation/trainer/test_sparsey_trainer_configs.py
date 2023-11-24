# -*- coding: utf-8 -*-

"""
Test Sparsey Trainer Configs: tests covering the config files for 
    customizing the structure of trainers for Sparsey models.
"""


import pytest

from sparsepy.cli.config_validation.validate_config import validate_config


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
        valid_schema = {
            'optimizer': {
                'name': 'sparsey'
            }, 
            'metrics': [
                {
                    'name': 'accuracy',
                    'save': True
                },
                {
                    'name': 'exact_match_retrieval'
                }
            ]
        }

        return valid_schema


    def perform_assertion(self, schema: dict, expected_value: bool) -> None:
        """
        Performs an assertion based on the return value
        of a call to validate_config.
        """
        assert validate_config(
            schema, 'trainer',
            'sparsey'
        )[1] == expected_value


    def test_valid_trainer_schema(self, sparsey_trainer_schema: dict) -> None:
        """
        Tests the config file validation for the case
        where the config file for a Sparsey model is fully
        valid.

        Args:
            sparsey_trainer_schema: a dict containing the valid
            sparsey trainer schema to be used for testing, passed in 
            via pytest's fixture functionality.
        """        
        self.perform_assertion(sparsey_trainer_schema, True)


    def test_missing_optimizer_name(self, sparsey_trainer_schema: dict) -> None:
        """
        Tests the config file validation for the case
        where the config file for a Sparsey model is fully
        valid.

        Args:
            sparsey_trainer_schema: a dict containing the valid
            sparsey trainer schema to be used for testing, passed in 
            via pytest's fixture functionality.
        """        
        del sparsey_trainer_schema['optimizer']['name']

        self.perform_assertion(sparsey_trainer_schema, False)


    def test_invalid_optimizer_name(self, sparsey_trainer_schema: dict) -> None:
        """
        Tests the config file validation for the case
        where the config file for a Sparsey model is fully
        valid.

        Args:
            sparsey_trainer_schema: a dict containing the valid
            sparsey trainer schema to be used for testing, passed in 
            via pytest's fixture functionality.
        """        
        sparsey_trainer_schema['optimizer']['name'] = 'invalid_name'

        self.perform_assertion(sparsey_trainer_schema, False)


    def test_missing_metrics(self, sparsey_trainer_schema: dict) -> None:
        """
        Tests the config file validation for the case
        where the config file for a Sparsey model is fully
        valid.

        Args:
            sparsey_trainer_schema: a dict containing the valid
            sparsey trainer schema to be used for testing, passed in 
            via pytest's fixture functionality.
        """        
        del sparsey_trainer_schema['metrics']

        self.perform_assertion(sparsey_trainer_schema, False)


    def test_no_listed_metrics(self, sparsey_trainer_schema: dict) -> None:
        """
        Tests the config file validation for the case
        where the config file for a Sparsey model is fully
        valid.

        Args:
            sparsey_trainer_schema: a dict containing the valid
            sparsey trainer schema to be used for testing, passed in 
            via pytest's fixture functionality.
        """        
        sparsey_trainer_schema['metrics'] = []

        self.perform_assertion(sparsey_trainer_schema, False)
