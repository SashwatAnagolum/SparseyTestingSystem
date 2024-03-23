# -*- coding: utf-8 -*-

"""
Basis Set Size Increase: file holding the BasisSetSizeIncreaseMetricSchema class.
"""


from schema import Schema, Or, Optional, And, Use

from sparseypy.cli.config_validation.saved_schemas.abs_schema import AbstractSchema
from sparseypy.core.metrics.metric_factory import MetricFactory


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
                'name': Schema('basis_set_size_increase', error="name must be 'basis_set_size_increase'"),
                Optional('save', default=False): Schema(bool, error="save must be a boolean value"),
                Optional('reduction', default=None): Schema(Or('mean', 'none', 'sum', error="reduction must be 'mean', 'none', or 'sum'"), error="Invalid reduction value"),
                Optional('best_value', default='min_by_layerwise_mean'): Schema(
                        And(
                            Use(MetricFactory.is_valid_comparision), True
                            ), error="best_value must be the name of a valid comparison function from comparisons.py")
            }, 
            ignore_extra_keys=True,
            error="Invalid configuration for basis_set_size_increase metric"
        )

        return config_schema