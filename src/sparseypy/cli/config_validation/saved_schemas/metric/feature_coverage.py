import typing

from schema import Schema, Or, Optional, And, Use, Const

from sparseypy.cli.config_validation.saved_schemas.abs_schema import AbstractSchema
from sparseypy.core.metrics.metric_factory import MetricFactory

class FeatureCoverageMetricSchema(AbstractSchema):
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
                'name': Schema('feature_coverage', error="name must be 'feature_coverage'"),
                Optional('save', default=False): Schema(bool, error="save must be a boolean value"),
                Optional('reduction', default='none'): Or(
                    'layerwise_mean', 'layerwise_sum', 'mean', 'sum', 'highest_layer',
                    'highest_layer_mean', 'none', None,
                    error="reduction must be 'none', 'sum', 'highest_layer', or 'mean'"
                ),
                Optional('best_value', default='max_by_layerwise_mean'): Schema(
                        And(
                            Const(Use(MetricFactory.is_valid_comparision), True)
                            ), error="best_value must be the name of a valid comparison function from comparisons.py"),
                Optional('params', default={}): {}
            }, 
            error="Invalid configuration for feature_coverage metric"
        )

        return config_schema