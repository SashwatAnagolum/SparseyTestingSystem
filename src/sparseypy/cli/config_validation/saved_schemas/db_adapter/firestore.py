# -*- coding: utf-8 -*-

"""
Firestore DB Adapter: file holding the FirestoreDbAdapterSchema class.
"""

import os
import typing

from schema import Schema, And, Optional, Use

from sparseypy.cli.config_validation.saved_schemas.abs_schema import AbstractSchema


class FirestoreDbAdapterSchema(AbstractSchema):
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
                'name': And(str, lambda n: n == 'firestore', error="name must be 'firestore'"),
                'firebase_service_key_path': And(Use(os.getenv), str, os.path.isfile, error="Firebase service account key file must exist"),
                'data_resolution': And(int, lambda x : 0 <= x <= 2, error="data_resolution must be 0 (nothing), 1 (summary), or 2 (every step)"),
                Optional('hpo_table_name', default="hpo_runs"): And(str, error="Invalid hpo_table_name in firestore configuration"),
                Optional('experiment_table_name', default="experiments"): And(str, error="Invalid experiment_table_name in firestore configuration")
            },
            error="Invalid configuration for firestore database adapter"
        )

        return config_schema

