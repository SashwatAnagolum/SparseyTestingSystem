# -*- coding: utf-8 -*-

"""
Sparsey Trainer Schema: the schema for Sparsey trainer config files.
"""


import typing

from schema import Schema, Optional, And

from ..abs_schema import AbstractSchema
from sparsepy.core import optimizers


class SparseyTrainerSchema(AbstractSchema):
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
                    i.capitalize()
                    for i in config_info['optimizer']['name'].split('_')
                ]
            )

            if optimizer_name not in dir(optimizers):
                return None
        except KeyError:
            return None

        return schema_params


    def transform_schema(self, config_info: dict) -> dict:
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
                'optimizer': {
                    'name': str
                },
                'metrics': And(
                    list, lambda x: len(x) > 0,
                    [
                        {
                            'name': str,
                            Optional('save', default=False): bool
                        }
                    ]
                )
            }
        )

        return config_schema
