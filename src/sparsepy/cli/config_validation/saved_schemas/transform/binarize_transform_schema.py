# -*- coding: utf-8 -*-

"""
Binarize Transform Schema: the schema for Bianrize transform configs.
"""


from typing import Optional

from schema import Schema, And

from ..abs_schema import AbstractSchema
from ...saved_schemas import schema_utils


class BinarizeTransformSchema(AbstractSchema):
    """
    BinarizeTransformSchema: schema for Binarize transforms.
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
                'transform_name': And(str, lambda x: x == 'binarize'),
                'binarize_threshold': And(
                    float,
                    lambda x: schema_utils.is_between(x, 0.0, 1.0)
                )
            }
        )

        return config_schema
