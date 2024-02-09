# -*- coding: utf-8 -*-

"""
Sparsey Trainer Schema: the schema for Sparsey trainer config files.
"""


import typing

from schema import Schema, Optional, And

from sparsepy.cli.config_validation.saved_schemas.abs_schema import AbstractSchema
from sparsepy.cli.config_validation.saved_schemas import optimizer
from sparsepy.cli.config_validation.saved_schemas import schema_utils
from sparsepy.cli.config_validation import schema_factory


class SparseyTrainingRecipeSchema(AbstractSchema):
    """
    SparseyTrainerSchema: schema for Sparsey trainers.
    """
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

        schema_params['optimizer_schema'] = []

        try:
            optimizer_name = ''.join(
                [
                    i for i in config_info['optimizer']['name'].split('_')
                ]
            )

            schema_params['optimizer_schema'] = schema_factory.get_schema_by_name(
                optimizer, 'optimizer', optimizer_name
            )
        except KeyError as e:
            print(e)
            return None

        return schema_params


    def transform_schema(self, config_info: dict) -> dict:
        #config_info['optimizer']['params'] = dict()
        
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
                    [
                        {
                            'name': str,
                            Optional('save', default=False): bool
                        }
                    ]
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
