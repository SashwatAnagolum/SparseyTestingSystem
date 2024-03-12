# -*- coding: utf-8 -*-

"""
Image dataset schema: the schema for Image dataset config files.
"""


import typing
import os

from schema import Schema, Optional, And

from sparsepy.cli.config_validation.saved_schemas.abs_schema import AbstractSchema
from sparsepy.core import optimizers


class BinarizeTransformSchema(AbstractSchema):
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
        config_schema = Schema(
            {
                'name': And(str, lambda n: n == 'binarize', error="Name must be 'binarize'."),
                'params': {
                    'binarize_threshold': And(float, lambda t: 0.0 <= t <= 1.0, error="Binarize threshold must be a float between 0.0 and 1.0 inclusive.")
                }
            }
        )

        return config_schema
