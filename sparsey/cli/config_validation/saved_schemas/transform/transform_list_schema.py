# -*- coding: utf-8 -*-

"""
Transform List Schema: the schema for transform list configs.
"""


from typing import Optional

from schema import Schema

from ..abs_schema import AbstractSchema
from ..transform import individual_transforms

class TransformListSchema(AbstractSchema):
    """
    TransformListSchema: schema for lists of transforms.
    """
    def __init__(self):
        """
        Initializes the TransformListSchema object by creating a set
        of all the accepted individual transforms that can be part
        of the transform list. This is done using reflection, and excluding
        all attributes of the individual_transforms module that start with 
        '__', which captures all built-in attributes.
        """
        self.allowed_transforms = [
            i for i in dir(individual_transforms) if i[:2] != '__'
        ]

        self.allowed_transforms = set(self.allowed_transforms)


    def get_transform_schema(self, transform_name: str) -> AbstractSchema:
        """
        Returns the transform schema corresponding to the name passed in.

        Args:
            transform_name: a str containing the name of the transform

        Returns:
        an AbstractSchema corresponding to the name of the transform
            passed in.
        """
        transform_module_name = f'{transform_name}_transform_schema'
        transform_class_name = f'{transform_name.capitalize()}TransformSchema'

        if transform_module_name not in dir(individual_transforms):
            raise ValueError(
                f'Invalid transform name {transform_module_name}!'
            )

        transform_module = getattr(
            individual_transforms, transform_module_name
        )

        transform_class = getattr(transform_module, transform_class_name)

        return transform_class()


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

        schema_params['transform_schemas'] = []

        for transform in config_info['transform_list']:
            transform_name = transform['transform_name']
            module_name = f'{transform_name}_transform_schema'

            if module_name not in self.allowed_transforms:
                return None

            transform_schema = self.get_transform_schema(transform_name)

            schema_params['transform_schemas'].append(
                transform_schema
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
        config_schema = Schema(
            {
                'transform_list': [
                    schema for schema in schema_params['transform_schemas']
                ]
            }
        )

        return config_schema
