# -*- coding: utf-8 -*-

"""
Schemas: functions to get schemas
"""

import os
import importlib

from schema import Schema

from validation_constants import allowed_schema_types, allowed_schema_names


def get_schema(schema_type: str, schema_name: str) -> Schema:
    """
    Gets and returns the schema corresponding the the passed in schema type
    and name.

    Args:
        schema_type: a str containing the type of schema to be used
        schema_name: a str containing the name of the schame to be used
    """
    if schema_type not in allowed_schema_types:
        raise ValueError(f'Invalid schema type: {schema_type}!')
    
    if schema_name not in allowed_schema_names[schema_type]:
        raise ValueError(f'Invalid schema name: {schema_name}!')

    schema_module = importlib.import_module(f'.{schema_type}.{schema_name}', 'saved_schemas')

    return schema_module.config_schema
