# -*- coding: utf-8 -*-

"""
Image dataset schema: the schema for Image dataset config files.
"""


import typing
import os

from schema import Schema, Optional, And

from sparsepy.cli.config_validation.saved_schemas.abs_schema import AbstractSchema
from sparsepy.cli.config_validation import schema_factory
from sparsepy.cli.config_validation.saved_schemas import preprocessing_stack


class ImageDatasetSchema(AbstractSchema):
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
                'dataset_type': Schema('image', error="dataset_type must be 'image'"),
                'params': Schema({
                    'data_dir': Schema(And(str, os.path.exists), error=f"Invalid data_dir path. The directory must exist."),
                    'image_format': Schema(And(str, lambda x: x[0] == '.'), error=f"Invalid image_format. The format must start with '.'")
                }, error="Invalid params"),
                Optional('preprocessed', default=False): Schema(bool, error="preprocessed must be a boolean value"),
                'preprocessed_stack': schema_params[
                    'preprocessing_stack_schema'
                ],
            }
        )

        return config_schema
