# -*- coding: utf-8 -*-

"""
Transform List Schema: the schema for transform list configs.
"""


from typing import Optional

from schema import Schema, SchemaError

from ..abs_schema import AbstractSchema
from .. import transform
from ... import schema_factory

class TransformListSchema(AbstractSchema):
    """
    TransformListSchema: schema for lists of transforms.
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

        if 'transform_list' not in config_info:
            return None

        schema_params['transforms'] = []

        for ind_transform in config_info['transform_list']:
            transform_name = ind_transform['transform_name']

            if transform_name not in dir(transform):
                return None

            schema_params['transforms'].append(
                transform_name
            )

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
        schema_list = []

        for transform_name in schema_params['transforms']:
            schema_list.append(
                schema_factory.get_schema_by_name(
                    transform, 'transform', transform_name
                )
            )

        return schema_list


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
           return None, False

        schema = self.build_schema(schema_params)

        valid_config_info = {'transform_list': []}

        try:
            for index, curr_transform in enumerate(
                config_info['transform_list']
            ):
                valid_transform_config = schema[index].validate(
                    curr_transform
                )

                valid_config_info['transform_list'].append(valid_transform_config)
        except SchemaError as e:
            print(e, '\n')

            return None, False

        return valid_config_info, True
