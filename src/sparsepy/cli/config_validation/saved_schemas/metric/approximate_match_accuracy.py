import typing

from schema import Schema, And, Optional

from sparsepy.cli.config_validation.saved_schemas.abs_schema import AbstractSchema
from sparsepy.cli.config_validation.saved_schemas import schema_utils

class ApproximateMatchAccuracyMetricSchema(AbstractSchema):

    def extract_schema_params(
            self, config_info: dict) -> typing.Optional[dict]:
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
                'name':'approximate_match_accuracy',
                Optional('save', default=False): bool,
                Optional('reduction', default=None): str
            }, ignore_extra_keys=True
        )

        return config_schema