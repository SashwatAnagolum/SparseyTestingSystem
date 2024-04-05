# -*- coding: utf-8 -*-

"""
Firestore DB Adapter: file holding the FirestoreDbAdapterSchema class.
"""

import os

from schema import Schema, And, Optional, Use

from sparseypy.cli.config_validation.saved_schemas.abs_schema import AbstractSchema


class FirestoreDbAdapterSchema(AbstractSchema):
    """
    FirestoreDbAdapterSchema: Schema class for the Firestore database adapter.
    """
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
                'name': And(
                    str,
                    lambda n: n == 'firestore',
                    error="name must be 'firestore'"
                ),
                'firebase_service_key_path': And(
                    Use(os.getenv),
                    str,
                    os.path.isfile,
                    error="Firebase service account key file must exist"
                ),
                'data_resolution': And(
                    int,
                    lambda x : 0 <= x <= 2,
                    error="data_resolution must be 0 (nothing), 1 (summary), or 2 (every step)"
                ),
                Optional('save_models', default=False): Schema(
                    bool,
                    error="save_models must be a Boolean value"
                ),
                Optional('batch_size', default=64): And(
                    int,
                    lambda x : x > 0,
                    error="batch_size must be a positive integer"
                ),
                Optional('table_names',
                        default={
                            v:v
                            for v in ["batches", "experiments", "hpo_runs",
                                    "model_registry", "models"]
                        }
                    ):
                        {
                            Optional("batches", default="batches"):
                                Schema(str, error="batches table name must be a string"),
                            Optional("experiments", default="experiments"):
                                Schema(str, error="experiments table name must be a string"),
                            Optional("hpo_runs", default="hpo_runs"):
                                Schema(str, error="hpo_runs table name must be a string"),
                            Optional("model_registry", default="model_registry"):
                                Schema(str, error="model_registry table name must be a string"),
                            Optional("models", default="models"):
                                Schema(str, error="models table name must be a string"),
                        }
            },
            error="Invalid configuration for firestore database adapter"
        )

        return config_schema
