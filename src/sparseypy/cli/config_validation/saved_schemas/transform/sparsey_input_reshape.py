# -*- coding: utf-8 -*-

"""
Sparsey Input Reshape schema: the schema for the SparseyInputReshape transform.
"""


from schema import Schema

from sparseypy.cli.config_validation.saved_schemas.abs_schema import AbstractSchema


class SparseyInputReshapeTransformSchema(AbstractSchema):
    """
    SparseyInputReshapeTransformSchema: schema for
    the Sparsey Input Reshape transform.
    """
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
                'name': 'sparsey_input_reshape',
                'params': {}
            }
        )

        return config_schema
