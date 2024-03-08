# -*- coding: utf-8 -*-

"""
Schema Factory: functions to get schemas
"""


from .saved_schemas.abs_schema import AbstractSchema


def get_schema_by_name(schema_module,
                       schema_type: str, schema_name: str) -> AbstractSchema:
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
    if schema_name not in dir(schema_module):
        raise ValueError(f'Invalid schema name: {schema_name}!')

    schema_module = getattr(schema_module, schema_name)

    #Very weird fix to naming convention here specific to transforms
    if (schema_type == 'transform'):
        schema_class_name = ''.join(
        [i.capitalize() for i in schema_name.split('_')] +
        ['Schema']
    )
    else:
        schema_class_name = ''.join(
            [i.capitalize() for i in schema_name.split('_')] +
            [i.capitalize() for i in schema_type.split('_')] +
            ['Schema']
        )

    return getattr(schema_module, schema_class_name)()
