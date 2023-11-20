# -*- coding: utf-8 -*-

"""
Abs Schema: file containing the base class for all Schemas.
"""


import abc

from typing import Optional

from schema import Schema, SchemaError


class AbstractSchema():
    """
    AbstractSchema: a base class for schemas. 
        All schemas are used to vwalidate different config files
        passed in by the user to define model structures, training 
        recipes, HPO runs, and create plots.
    """
    @abc.abstractmethod
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


    @abc.abstractmethod
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


    def validate(self, config_info: dict) -> Optional[dict]:
        """
        Validates a given configuration against the 
        schema defined by the class.

        Args:
            config_info: a dict containing all of the configuration
                information passed in by the user.
            schema: a Schema to be used for validation

        Returns:
            a dict (might be None) holding the validated
                (and possibly transformed) user config info.
        """
        schema_params = self.extract_schema_params(config_info)

        if schema_params is None:
            raise ValueError('The passed in configuration file is invalid!')

        schema = self.build_schema(schema_params)

        try:
            valid_config_info = schema.validate(
                config_info
            )

            return valid_config_info
        except SchemaError as e:
            print(e, '\n')

            return None
