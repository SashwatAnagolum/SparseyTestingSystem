# -*- coding: utf-8 -*-

"""
Basis Average: file holding the BasisAverageMetricSchema class.
"""


import typing

from schema import Schema, And, Optional

from sparsepy.cli.config_validation.saved_schemas.abs_schema import AbstractSchema


class BasisAverageMetricSchema(AbstractSchema):
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
                'name': 'basis_average',
                Optional('save', default=False): bool,
                Optional('reduction', default=None): 'sparse'
            }
        )

        return config_schema
