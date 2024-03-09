# -*- coding: utf-8 -*-

"""
Sparsey Trainer Schema: the schema for Sparsey trainer config files.
"""


import typing

from schema import Schema, Optional, And, Or

from sparsepy.cli.config_validation.saved_schemas.abs_schema import AbstractSchema
from sparsepy.cli.config_validation import schema_factory
from sparsepy.cli.config_validation.saved_schemas import (
    schema_utils, metric, optimizer
)


class SparseyTrainingRecipeSchema(AbstractSchema):
    """
    SparseyTrainerSchema: schema for Sparsey trainers.
    """
    def check_if_optimizer_exists(self, optimizer_name) -> bool:
        """
        Checks if the optimizer with optimizer_name exists or not.

        Args:
            optimizer_name (str): the name of the optimizer.

        Returns:
            (bool): whether the optimizer exists or not.
        """
        try:
            schema_factory.get_schema_by_name(
                optimizer, 'optimizer', optimizer_name
            )
        except ValueError:
            return False

        return True


    def check_if_metric_exists(self, metric_name) -> bool:
        """
        Checks if a metric exists or not.

        Returns:
            (bool): whether the metric exists or not.
        """
        try:
            schema_factory.get_schema_by_name(
                metric, 'metric', metric_name
            )
        except ValueError:
            return False

        return True


    def build_precheck_schema(self) -> Schema:
        """
        Builds the precheck schema for the config information
        passed in by the user. This is used to verify that all parameters
        can be collected in order to build the actual schema that will
        be used to verify the entire configuration passed in by the
        user.

        Returns:
            (Schema): the precheck schema.
        """
        return Schema(
            {
                'optimizer': {
                    'name': self.check_if_optimizer_exists
                },
                'metrics': [
                    {
                        'name': self.check_if_metric_exists
                    }
                ]
            }, ignore_extra_keys=True
        )


    def extract_schema_params(
            self, config_info: dict) -> typing.Optional[dict]:
        """
        Extracts the required schema parameters from the config info dict
        in order to build the schema to validate against.

        Args:
            config_info: a dict containing the config info from the 
                user.

        Returns:
            a dict (might be None) containing all the required parameters 
                to build the schema.
        """
        schema_params = dict()

        schema_params['optimizer_schema'] = schema_factory.get_schema_by_name(
            optimizer, 'optimizer', config_info['optimizer']['name']
        )

        schema_params['metric_schemas'] = []

        for metric_info in config_info['metrics']:
            schema_params['metric_schemas'].append(
                schema_factory.get_schema_by_name(
                    metric, 'metric', metric_info['name']
                )
            )

        return schema_params


    def transform_schema(self, config_info: dict) -> dict:
        """
        Transforms the config info passed in by the user to 
        construct the config information required by the model builder.

        Args:
            config_info: dict containing the config information

        Returns:
            (dict): the transformed config info
        """
        config_info['optimizer']['params'] = dict()
        
        return config_info


    def build_schema(self, schema_params: dict) -> Schema:
        """
        Builds a schema that can be used to validate the passed in
        config info.

        Args:
            schema_params: a dict containing all the required
                parameters to build the schema.

        Returns:
            a Schema that can be used to validate the config info.
        """
        config_schema = Schema(
            {
                'optimizer': schema_params['optimizer_schema'],
                'metrics': And(
                    list, lambda x: len(x) > 0,
                    [Or(*schema_params['metric_schemas'])]
                ),
                'dataloader': {
                    'batch_size': And(int, schema_utils.is_positive),
                    'shuffle': bool
                },
                'training': {
                    'num_epochs': And(int, schema_utils.is_positive),
                    Optional('step_resolution', default=None): And(
                        int, schema_utils.is_positive
                    )
                }
            }
        )

        return config_schema
