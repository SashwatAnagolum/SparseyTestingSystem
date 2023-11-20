# -*- coding: utf-8 -*-

"""
Schemas: functions to get schemas
"""


from . import saved_schemas
from .saved_schemas.abs_schema import AbstractSchema


def get_schema_by_name(schema_type: str, schema_name: str) -> AbstractSchema:
    """
    Gets and returns the schema corresponding the the passed in schema type
    and name.

    Args:
        schema_type: a str containing the type of schema to be used
        schema_name: a str containing the name of the schame to be used

    Returns:
        an AbstractSchema corresponding to the name and type
            passed in.
    """
    if schema_type not in dir(saved_schemas):
        raise ValueError(f'Invalid schema type: {schema_type}!')

    schema_type_module = getattr(saved_schemas, schema_type)
    schema_name_module = schema_name + '_schema'

    if schema_name_module not in dir(schema_type_module):
        raise ValueError(f'Invalid schema name: {schema_name}!')

    schema_module = getattr(schema_type_module, schema_name_module)

    schema_name_parts = schema_name.split('_')
    schema_class_name = ''.join(
        [i.capitalize() for i in schema_name_parts] + ['Schema']
    )

    return getattr(schema_module, schema_class_name)
