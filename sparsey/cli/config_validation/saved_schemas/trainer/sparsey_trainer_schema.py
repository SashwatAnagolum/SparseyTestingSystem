# -*- coding: utf-8 -*-

"""
Sparsey Trainer Schema: the schema for Sparsey trainer config files.
"""


from typing import Optional

from schema import Schema, And

from ..abs_schema import AbstractSchema


class SparseyTrainerSchema(AbstractSchema):
    """
    SparseyTrainerSchema: schema for Sparsey trainers.
    """
    def extract_schema_params(self, config_info: dict) -> Optional[dict]:
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

        return schema_params


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
                'weight_update_rule': And(str, lambda x: x in ['hebbian'])
            }
        )

        return config_schema
