from schema import Schema, Or, Optional, And, Use, Const

from sparseypy.cli.config_validation.saved_schemas.abs_schema import AbstractSchema
from sparseypy.core.metrics.metric_factory import MetricFactory


class BasisSetSizeMetricSchema(AbstractSchema):
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
                'name': Schema('basis_set_size', error="name must be 'basis_set_size'"),
                Optional('save', default=False): Schema(bool, error="save must be a boolean value"),
                Optional('reduction', default='none'): Or(
                    'none', 'mean', 'sum', 'highest_layer',
                    error="reduction must be 'none', 'mean', 'highest_layer', or 'sum'"
                ),
                Optional('best_value', default='min_by_layerwise_mean'): Schema(
                        And(
                            Const(Use(MetricFactory.is_valid_comparision), True)
                            ), error="best_value must be the name of a valid comparison function from comparisons.py"),
                Optional('params', default={}): {}
            },
            ignore_extra_keys=True,
            error="Invalid configuration for basis_set_size metric"
        )

        return config_schema