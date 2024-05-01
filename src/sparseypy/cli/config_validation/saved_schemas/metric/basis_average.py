# -*- coding: utf-8 -*-

"""
Basis Average: file holding the BasisAverageMetricSchema class.
"""


from schema import Schema, And, Optional, Or, Use, Const

from sparseypy.cli.config_validation.saved_schemas.abs_schema import AbstractSchema
from sparseypy.core.metrics.metric_factory import MetricFactory


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
                'name': And(str, lambda n: n == 'basis_average', error="name must be 'basis_average'"),
                Optional('save', default=False): And(bool, error="save must be a boolean value"),
                Optional('reduction', default='none'): Or(
                    'sparse', None, 'none', 'highest_layer',
                    error="Reduction must be one of 'sparse'"
                        ", 'none', or 'highest_layer'"
                ),
                Optional('best_value', default='max_by_layerwise_mean'): Schema(
                    And(
                        Const(Use(MetricFactory.is_valid_comparision), True)
                    ), error="best_value must be the name of a valid comparison function from comparisons.py"
                ),
                Optional('params', default={}): {}
            },
            error="Invalid configuration for basis_average metric"
        )

        return config_schema
