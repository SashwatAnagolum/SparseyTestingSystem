import typing

from schema import Schema, Or, Optional, Use, And

from sparsepy.cli.config_validation.saved_schemas.abs_schema import AbstractSchema
from sparsepy.core.metrics.metric_factory import MetricFactory

class MatchAccuracyMetricSchema(AbstractSchema):
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
                'name': Schema('match_accuracy', error="name must be 'match_accuracy'"),
                Optional('save', default=False): Schema(bool, error="save must be a boolean value"),
                Optional('reduction', default=None): Schema(Or('none', 'mean', 'sum', error="reduction must be 'none', 'mean', or 'sum'"), error="Invalid reduction value"),
                Optional('best_value', default='max_by_layerwise_mean'): Schema(
                        And(
                            Use(MetricFactory.is_valid_comparision), True
                            ), error="best_value must be the name of a valid comparison function from comparisons.py")
            }, 
            ignore_extra_keys=True,
            error="Invalid configuration for match_accuracy metric"
        )

        return config_schema