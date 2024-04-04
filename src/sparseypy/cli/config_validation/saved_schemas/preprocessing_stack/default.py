# -*- coding: utf-8 -*-

"""
Default Preprocessing stack schema: file holding the schema for default 
    preprocessing stacks.
"""


import typing

from schema import Schema, And, Or, Use
from torchvision.transforms import v2

from sparseypy.cli.config_validation.saved_schemas.abs_schema import AbstractSchema
from sparseypy.cli.config_validation import schema_factory
from sparseypy.cli.config_validation.saved_schemas import (
    model, schema_utils, metric, transform
)


class DefaultPreprocessingStackSchema(AbstractSchema):
    """
    Default Preprocessing Stack Schema: class for preprocessing stack schemas.
    """
    def check_if_transform_exists(self, transform_name) -> bool:
        """
        Checks if a model family with the name model_family exists.

        Args:
            transform_name (str): the name of the transform to check.

        Returns:
            (bool): whether the model famly exists or not
        """
        if not schema_factory.schema_exists_by_name(
                transform, 'transform', transform_name
            ):
            converted_transform_name = ''.join(
                [word[:1].upper() + word[1:] for word in transform_name.split('_')]
            )

            if not hasattr(v2, converted_transform_name):
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
                'transform_list': [
                    {
                        'name': And(
                            Use(str),
                            self.check_if_transform_exists,
                            error="Transform name does not exist in the defined schemas or torchvision."
                        )
                    }
                ]
            }, ignore_extra_keys=True
        )


    def extract_schema_params(self, config_info: dict) -> dict:
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

        schema_params['transform_schemas'] = []

        for transform_info in config_info['transform_list']:
            if schema_factory.schema_exists_by_name(
                transform, 'transform', transform_info['name']
            ):
                transform_schema = schema_factory.get_schema_by_name(
                    transform, 'transform', transform_info['name']
                )
            else:
                transform_schema = Schema(object)

            schema_params['transform_schemas'].append(
                transform_schema
            )

        return schema_params


    def check_transform_schema_validity(
        self, transform_configs: dict,
        transform_schemas: list[Schema]) -> bool:
        """
        Checks if all of the transform config information
        if valid or not.

        Args:
            transform_configs (dict): the transform config
                information to check
            transofmr_schemas (list[Schema]): the schemas
                to validate against.

        Returns:
            (bool): whether all of the transform configs
                are valid or not.
        """
        for i in range(len(transform_configs)):
            transform_schemas[i].validate(transform_configs[i])

        return True


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
                'transform_list': And(
                    [dict],  # Ensures the input is a list of dictionaries
                    Use(list),  # Ensures the input is processed as a list
                    lambda x: self.check_transform_schema_validity(
                        x, schema_params['transform_schemas']
                    ),
                    error="Invalid configuration for one or more transforms in the transform list."
                )
            }
        )
        
        return config_schema
