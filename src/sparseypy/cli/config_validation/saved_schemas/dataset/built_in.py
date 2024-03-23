# -*- coding: utf-8 -*-

"""
Named dataset schema: the schema for named dataset config files.

Does NOT correspond to a NamedDataset class.
"""


import typing
import os

from schema import Schema, Optional, And, Use, Const

from sparseypy.cli.config_validation.saved_schemas.abs_schema import AbstractSchema
from sparseypy.cli.config_validation import schema_factory
from sparseypy.cli.config_validation.saved_schemas import preprocessing_stack
from sparseypy.access_objects.datasets.dataset_factory import DatasetFactory

class BuiltInDatasetSchema(AbstractSchema):
    """
    BuiltInDatasetSchema: schema for built-in datasets (created by name, e.g. MNIST).
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
                'dataset_type': Schema('built_in', error="dataset_type must be 'built_in'"),
                'params': Schema({
                    'name': Schema(And(str, And(Const(Use(DatasetFactory.is_valid_builtin_dataset), True))), error="The dataset name is invalid or does not exist."),
                    'data_dir': Schema(And(str, os.path.exists), error="The path at which the built-in dataset will be saved must exist."),
                    Optional('download', default=True): bool
                }, error="Invalid params"),
                Optional('preprocessed', default=False): Schema(bool, error="preprocessed must be a boolean value"),
                Optional('preprocessed_temp_dir', default='datasets/preprocessed_dataset'): Schema(str, error="preprocessed_temp_dir must be a valid path"),
                'preprocessed_stack': schema_params[
                    'preprocessing_stack_schema'
                ],
            }, error="Error in built-in dataset configuration"
        )

        return config_schema