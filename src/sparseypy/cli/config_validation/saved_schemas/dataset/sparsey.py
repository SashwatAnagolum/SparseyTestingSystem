# -*- coding: utf-8 -*-

"""
Image dataset schema: the schema for Image dataset config files.
"""


import typing
import os

from schema import Schema, Optional, And

from sparseypy.cli.config_validation.saved_schemas.abs_schema import AbstractSchema
from sparseypy.cli.config_validation import schema_factory
from sparseypy.cli.config_validation.saved_schemas import preprocessing_stack


class SparseyDatasetSchema(AbstractSchema):
    """
    SparseyDatasetSchema: schema for a Sparsey-style raw binary dataset.
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

        if config_info.get('preprocessed', False):
            schema_params[
                'preprocessing_stack_schema'
            ] = schema_factory.get_schema_by_name(
                preprocessing_stack, 'preprocessing_stack',
                'default'
            )
        else:
            schema_params[
                'preprocessing_stack_schema'
            ] = Schema(object)

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
                'dataset_type': Schema('sparsey', error="dataset_type must be 'sparsey'"),
                Optional('description', default=None): str,
                'params': Schema({
                    'data_dir': Schema(
                                    And(str, os.path.exists), 
                                    error="Invalid data_dir path. The directory must exist."),
                    'width': And(int, lambda x : x > 0, error="Width must be a positive integer"),
                    'height': And(int, lambda x : x > 0, error="Height must be a positive integer")
                }, error="Invalid params"),
                Optional('preprocessed', default=False): 
                    Schema(bool, error="preprocessed must be a boolean value"),
                Optional('preprocessed_temp_dir', default='datasets/preprocessed_dataset'): 
                    Schema(str, error="preprocessed_temp_dir must be a valid path"),
                Optional(
                    'preprocessed_stack',
                    default={'transform_list': []}
                ): schema_params['preprocessing_stack_schema'],
                Optional('save_to_disk', default=False): Schema(
                    bool, error='save_to_disk must be a boolean value'
                ),
                Optional('in_memory', default=False): Schema(
                    bool, "in_memory must be a boolean value"
                ),
                Optional('load_lazily', default=True): Schema(
                    bool, "load_lazily must be a boolean value"
                )
            }, error="Invalid Sparsey binary dataset configuration"
        )

        return config_schema
