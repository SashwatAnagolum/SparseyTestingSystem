# -*- coding: utf-8 -*-

"""
Num Activations: file holding the NumActivationsMetricSchema class.
"""

import typing

from schema import Schema, Or, Optional, And, Use

from sparsepy.cli.config_validation.saved_schemas.abs_schema import AbstractSchema
from sparsepy.core.metrics.metric_factory import MetricFactory

class NumActivationsMetricSchema(AbstractSchema):
    def extract_schema_params(self, config_info: dict) -> typing.Optional[dict]:
        """
        Extracts the required schema parameters from the config info dict
        in order to build the schema to validate against.

        In this instance, there are no parameters.

        Args:
            config_info: a dict containing the config info from the
                user.

        Returns:
            a dict (might be None) containing all the required parameters
                to build the schema.
        """
        schema_params = dict()

        return schema_params


    def transform_schema(self, config_info: dict) -> dict:
        return config_info


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
                'name': Schema('num_activations', error="name must be 'num_activations'"),
                Optional('save', default=False): Schema(bool, error="save must be a boolean value"),
                Optional('reduction', default=None): Schema(Or(
                    'none', 'layerwise_mean', 'sum', 'mean', error="reduction must be 'none', 'layerwise_mean', 'sum', or 'mean'"
                ), error="Invalid reduction value"),
                Optional('best_value', default='max_by_layerwise_mean'): Schema(
                        And(
                            Use(MetricFactory.is_valid_comparision), True
                            ), error="best_value must be the name of a valid comparison function from comparisons.py")
            },
            ignore_extra_keys=True,
            error="Invalid configuration for num_activations metric"
        )

        return config_schema
