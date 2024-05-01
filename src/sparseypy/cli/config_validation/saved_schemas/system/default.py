# -*- coding: utf-8 -*-

"""
Default System Schema: the schema for system.yaml.
"""

import os

from schema import Schema, And, Or, Optional, Use

from sparseypy.cli.config_validation.saved_schemas.abs_schema import AbstractSchema
from sparseypy.cli.config_validation import schema_factory
from sparseypy.cli.config_validation.saved_schemas import (
    db_adapter
)


class DefaultSystemSchema(AbstractSchema):
    """
    Default System Schema: class for system.yaml schema.
    """
    def build_precheck_schema(self) -> Schema:
        """
        Builds the precheck schema for the config information
        passed in by the user. This is used to verify that all parameters
        can be collected in order to build the actual schema that will
        be used to verify the entire configuration passed in by the
        user.

        Returns:
            (Schema): the precheck schema.
        """
        return Schema(
            {
                'database': {
                    'read_database': self.check_if_db_adapter_exists,
                    'write_databases': [
                        {
                            'name': self.check_if_db_adapter_exists
                        }
                    ]
                }
            }, ignore_extra_keys=True
        )


    def check_if_db_adapter_exists(self, db_adapter_name) -> bool:
        """
        Checks if a database adapter exists or not.

        Returns:
            (bool): whether the database adapter exists or not.
        """
        return schema_factory.schema_exists_by_name(
                db_adapter, 'db_adapter', db_adapter_name
            )


    def extract_schema_params(self, config_info: dict) -> dict:
        """
        Extracts the required schema parameters from the config info dict
        in order to build the schema to validate against.

        Args:
            config_info: a dict containing the config info from the 
                user.

        Returns:
            a dict (might be None) containing all the required parameters 
                to build the schema.
        """
        schema_params = dict()

        schema_params['database_schemas'] = []
        schema_params['selected_dbs'] = []

        for db_info in config_info['database']['write_databases']:
            schema_params['database_schemas'].append(
                schema_factory.get_schema_by_name(
                    db_adapter, 'db_adapter', db_info['name']
                )
            )

            schema_params['selected_dbs'].append(
                db_info['name']
            )

        return schema_params

    def make_env_schema(self, env_name: str):
        """
        Builds a schema that can be used to validate a string that is either
        the name of an environment variable (with $ prefix) or a value.

        Args:
            env_name (str): the value or environment variable name

        Returns:
            the value or a Use that can be used to validate the value
        """
        if env_name[0] == "$":
            return Schema(Use(os.getenv), str).validate(env_name)
        else:
            return env_name

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
                    'console': Schema(
                        {
                            Optional('hpo_progress_bars', default=True):
                                Schema(
                                    bool,
                                    error="hpo_progress_bars must be a Boolean value"
                                ),
                            Optional('print_best_hpo_config', default=False):
                                Schema(
                                    bool,
                                    error="print_best_hpo_config must be a Boolean value"
                                ),
                            Optional('print_error_stacktrace', default=False):
                                Schema(
                                    bool,
                                    error="print_error_stacktrace must be a Boolean value"
                                ),
                            Optional('print_metric_values', default=False):
                                Schema(
                                    bool,
                                    error="print_metric_values must be a Boolean value"
                                )
                        }
                    ),
                    'wandb': Schema({
                        Optional('api_key', default="WANDB_API_KEY"):
                            And(
                                Use(os.getenv),
                                str,
                                error="Invalid Weights and Biases API key"
                            ),
                        'project_name': 
                            Schema(str, error="Project name must be a string"),
                        Optional('data_resolution', default=2):
                            And(
                                int,
                                lambda x : 0 <= x <= 2,
                                error="data_resolution must be 0, 1, or 2"
                            ),
                        Optional('entity', default=None):
                            Schema(str, error="Entity name must be a string"),
                        Optional('local_log_directory', default=None):
                            Schema(str, error="local_log_directory must be a string"),
                        Optional('remove_local_files', default=False):
                            Schema(bool, error="remove_local_files must be a Boolean value"),
                        Optional('save_locally', default=True):
                            Schema(bool, error="save_locally must be a Boolean value"),
                        Optional('save_models', default=True):
                            Schema(bool, error="save_models must be a Boolean value"),
                        Optional('silent', default=True):
                            Schema(bool, error="silent must be a Boolean value")
                    },
                    error="Error in wandb configuration"),
                    'database': Schema({
                        'read_database': 
                            Schema(
                                lambda x : x in schema_params['selected_dbs'],
                                error="The read_database must also be chosen as a write_database"
                            ),
                        'write_databases': 
                            [
                                Or(*schema_params['database_schemas'],
                                error="Invalid database configuration schema")
                            ]
                    },
                    error="Error in database configuration"),
                },
                error="Error in system.yaml"
        )

        return config_schema
