# -*- coding: utf-8 -*-

"""
Transform List Schema: the schema for transform list configs.
"""

from typing import Optional, Tuple
from schema import SchemaError
from ... import saved_schemas

from sparsepy.cli.config_validation.saved_schemas.abs_schema import AbstractSchema
from sparsepy.cli.config_validation.saved_schemas import transform
from sparsepy.cli.config_validation import schema_factory

# Importing PyTorch to check if a transform is a built-in PyTorch transform
import torchvision.transforms.v2 as torch_transforms


class PreprocessingStackSchema(AbstractSchema):
    """
    TransformListSchema: schema for lists of transforms.
    """
    def is_builtin_transform(self, transform_name: str) -> bool:
        """
        Checks if the given transform name corresponds to a built-in PyTorch transform.

        Args:
            transform_name: Name of the transform.

        Returns:
            True if it is a built-in PyTorch transform, False otherwise.
        """
        return hasattr(torch_transforms, transform_name.capitalize())

    def extract_schema_params(self, config_info: dict) -> Optional[dict]:
        """
        Extracts the required schema parameters from the config info dict
        in order to build the schema to validate against.

        Args:
            config_info: a dict containing the config info from the user.

        Returns:
            a dict (might be None) containing all the required parameters to build the schema.
        """
        if 'transform_list' not in config_info:
            return None

        schema_params = {'transforms': []}

        for ind_transform in config_info['transform_list']:
            transform_name = ind_transform['name']
            if (not self.is_builtin_transform(transform_name)) and (transform_name not in dir(transform)):
                raise ValueError("Invalid transform on transform list.")

            schema_params['transforms'].append(transform_name)

        return schema_params

    def build_schema(self, schema_params: dict) -> list[Optional[AbstractSchema]]:
        """
        Builds a schema that can be used to validate the passed in config info.

        Args:
            schema_params: a dict containing all the required parameters to build the schema.

        Returns:
            a list of Schemas or None for built-in transforms.
        """
        try:
            schema_module = getattr(saved_schemas, 'transform')
        except Exception:
            raise ValueError(f'Invalid schema type transform!')

        schema_list = []

        for transform_name in schema_params['transforms']:
            if self.is_builtin_transform(transform_name):
                schema_list.append(None)
            else:
                schema_list.append(
                    #schema_module = getattr(saved_schemas, 'transform')
                    schema_factory.get_schema_by_name(
                        schema_module, 'transform', transform_name
                    )
                )

        return schema_list

    def validate(self, config_info: dict) -> Tuple[Optional[dict], bool]:
        """
        Validates a given configuration against the schema defined by the class.

        Args:
            config_info: a dict containing all of the configuration information passed in by the user.

        Returns:
            a dict (might be None) holding the validated (and possibly transformed) user config info.
        """
        schema_params = self.extract_schema_params(config_info)

        if schema_params is None:
            return None, False

        schema = self.build_schema(schema_params)

        valid_config_info = {'transform_list': []}

        try:
            for index, curr_transform in enumerate(config_info['transform_list']):
                if schema[index] is not None:
                    valid_transform_config = schema[index].validate(curr_transform)
                else:
                    # Assuming built-in transforms are valid
                    valid_transform_config = curr_transform

                valid_config_info['transform_list'].append(valid_transform_config)

        except SchemaError as e:
            print(e, '\n')
            return None, False

        return valid_config_info, True
