# -*- coding: utf-8 -*-

"""
RandomHPOStrategySchema: the schema for RandomHPOStrategy.
"""


import typing

from schema import Schema

from sparsepy.cli.config_validation.saved_schemas.abs_schema import AbstractSchema


class RandomHpoStrategySchema(AbstractSchema):
    """
    Random HPO Strategy Schema: class for Random HPO Stategy schemas.
    """
    def extract_schema_params(self, config_info: dict) -> typing.Optional[
        dict
    ]:
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


    def transform_schema(self, config_info: dict) -> dict:
        """
        Transforms the config info passed in by the user to 
        construct the config information required to build the HPORun.

        Args:
            config_info: dict containing the config information

        Returns:
            dict containing the transformed config info
        """
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
        config_schema = Schema(None)

        return config_schema
