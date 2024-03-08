# -*- coding: utf-8 -*-

"""
Image dataset schema: the schema for Image dataset config files.
"""


import typing
import os

from schema import Schema, Optional, And
from sparsepy.cli.config_validation.saved_schemas.transform.preprocessing_stack_schema import PreprocessingStackSchemaTransformSchema
from sparsepy.cli.config_validation.saved_schemas.abs_schema import AbstractSchema
from sparsepy.core import optimizers


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
        def validate_preprocessed_stack(ps_config):
            # Check if 'preprocessed' is True in the config
            #if config.get('preprocessed', True):
                # If True, validate preprocessed_stack using PreprocessingStackSchemaTransformSchema
                
                #return PreprocessingStackSchemaTransformSchema.preprocessing_stack_schema.validate(config.get('preprocessed_stack', {}))
                #return PreprocessingStackSchemaTransformSchema.build_schema(config.get('preprocessed_stack', {}))
            preprocessing_schema = PreprocessingStackSchemaTransformSchema()
                #ps_config = config.get('preprocessed_stack', {})
            validated_config, is_valid = preprocessing_schema.validate(ps_config)
            if not is_valid:
                raise ValueError("Preprocessed stack configuration is invalid.")
            return True
            #else:
                # If preprocessed is False or not set, skip validation for preprocessed_stack
                #return True
            #return PreprocessingStackSchemaTransformSchema.validate(config.get('preprocessed_stack', {}))
            
        config_schema = Schema(
            {
                'dataset_type': 'image',
                'params': {
                    'data_dir': And(str, os.path.exists),
                    'image_format': And(str, lambda x: x[0] == '.')
                },
                'preprocessed': bool,
                'preprocessed_stack': validate_preprocessed_stack,
            }
        )

        return config_schema
