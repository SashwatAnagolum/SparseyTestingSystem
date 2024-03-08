# -*- coding: utf-8 -*-

"""
Abs Schema: file containing the base class for all Schemas.
"""


import abc
import sys

from typing import Optional

from schema import Schema


class AbstractSchema():
    """
    AbstractSchema: a base class for schemas. 
        All schemas are used to vwalidate different config files
        passed in by the user to define model structures, training 
        recipes, HPO runs, and create plots.
    """
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
        return Schema(object)


    def extract_schema_params(self, config_info: dict) -> dict:
        """
        Extracts the required schema parameters from the config info dict
        in order to build the schema to validate against.

        Args:
            config_info: a dict containing the config info from the 
                user.

        Returns:
            (dict): all the required parameters 
                to build the schema.
        """
        return dict()


    def transform_schema(self, config_info: dict) -> dict:
        """
        Transforms the config info passed in by the user to 
        construct the config information required by the model builder.

        Args:
            config_info: dict containing the config information

        Returns:
            (dict): the transformed config info
        """
        return config_info


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
        precheck_schema = self.build_precheck_schema()
        precheck_schema.validate(config_info)
        schema_params = self.extract_schema_params(config_info)
        schema = self.build_schema(schema_params)
        validated_config = schema.validate(config_info)

        return self.transform_schema(validated_config)
