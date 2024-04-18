# -*- coding: utf-8 -*-

"""
Sisc Adherence: file holding the SiscAdherenceMetricSchema class.
"""


from schema import Schema, And, Optional, Or, Use, Const

from sparseypy.cli.config_validation.saved_schemas.abs_schema import AbstractSchema
from sparseypy.core.metrics.metric_factory import MetricFactory


class SiscAdherenceMetricSchema(AbstractSchema):
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
                'name': And(str, lambda n: n == 'code_similarity', error="name must be 'code_similarity'"),
                Optional('save', default=False): And(bool, error="save must be a boolean value"),
                Optional('reduction', default='none'): Or(
                    'none', None, 'layerwise_mean', 'sum', 'mean', 'highest_layer',
                    error="reduction must be 'none', 'layerwise_mean', 'sum', 'highest_layer', or 'mean'"
                ),
                Optional('best_value', default='max_by_layerwise_mean'): Schema(
                    And(
                        Const(Use(MetricFactory.is_valid_comparision), True)
                    ), error="best_value must be the name of a valid comparison function from comparisons.py"
                )
            },
            error="Invalid configuration for sisc_adherence metric"
        )

        return config_schema
