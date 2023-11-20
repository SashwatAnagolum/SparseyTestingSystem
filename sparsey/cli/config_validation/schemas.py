# -*- coding: utf-8 -*-

"""
Schemas: functions to get schemas
"""


from . import saved_schemas
from .saved_schemas.abs_schema import AbstractSchema
from .validation_constants import allowed_schema_types, allowed_schema_names


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
    if schema_type not in allowed_schema_types:
        raise ValueError(f'Invalid schema type: {schema_type}!')

    if schema_name not in allowed_schema_names[schema_type]:
        raise ValueError(f'Invalid schema name: {schema_name}!')

    schema_type_module = getattr(saved_schemas, schema_type)
    schema_module = getattr(schema_type_module, schema_name)

    return getattr(schema_module, f'{schema_name.capitalize()}Schema')
