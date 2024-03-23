# -*- coding: utf-8 -*-

"""
Hebbian Optimizer Schema: the schema for Sparsey trainer config files.
"""


import typing

from schema import Schema, Optional, And, Use

from sparseypy.cli.config_validation.saved_schemas.abs_schema import AbstractSchema


class HebbianOptimizerSchema(AbstractSchema):
    """
    HebbianOptimizerSchema: schema for hebbian optimizers.
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

        params = config_info.get('params',{})
        #thresh = config_info.get('params', {}).get('thresh')
        #schema_params = {'thresh': thresh}

        return {'params': params}


    def transform_schema(self, config_info: dict) -> dict:
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
        optimizer_params_schema = {
            Optional('thresh', default=None): And(Use(float), lambda t: 0.0 <= t <= 1.0, error="thresh must be a float between 0.0 and 1.0 inclusive")
        }

        config_schema = Schema(
            {
                'name': Schema('hebbian', error="name must be 'hebbian'"),
                Optional('params'): Schema(optimizer_params_schema, error="Invalid params configuration for Hebbian optimizer")
            }, ignore_extra_keys=True
        )

        return config_schema
