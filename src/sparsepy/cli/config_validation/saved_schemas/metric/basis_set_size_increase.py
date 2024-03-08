# -*- coding: utf-8 -*-

"""
Basis Set Size Increase: file holding the BasisSetSizeIncreaseMetricSchema class.
"""


from schema import Schema, Or, Optional

from sparsepy.cli.config_validation.saved_schemas.abs_schema import AbstractSchema


class BasisSetSizeIncreaseMetricSchema(AbstractSchema):
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
                'name':'basis_set_size_increase',
                Optional('save', default=False): bool,
                Optional('reduction', default=None): Or('mean', 'none', 'sum')
            }, ignore_extra_keys=True
        )

        return config_schema