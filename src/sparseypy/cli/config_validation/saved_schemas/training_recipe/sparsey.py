# -*- coding: utf-8 -*-

"""
Sparsey Trainer Schema: the schema for Sparsey trainer config files.
"""


import typing

from schema import Schema, Optional, And, Use, Or

import torch

from sparseypy.cli.config_validation.saved_schemas.abs_schema import AbstractSchema
from sparseypy.cli.config_validation import schema_factory
from sparseypy.cli.config_validation.saved_schemas import (
    schema_utils, metric, optimizer
)


class SparseyTrainingRecipeSchema(AbstractSchema):
    """
    SparseyTrainerSchema: schema for Sparsey trainers.
    """
    def check_if_optimizer_exists(self, optimizer_name) -> bool:
        """
        Checks if the optimizer with optimizer_name exists or not.

        Args:
            optimizer_name (str): the name of the optimizer.

        Returns:
            (bool): whether the optimizer exists or not.
        """
        return schema_factory.schema_exists_by_name(
                    optimizer, 'optimizer', optimizer_name
                )


    def check_if_metric_exists(self, metric_name) -> bool:
        """
        Checks if a metric exists or not.

        Returns:
            (bool): whether the metric exists or not.
        """
        return schema_factory.schema_exists_by_name(
            metric, 'metric', metric_name
        )


    def check_if_gpu_exists(self):
        """
        Checks if a supported GPU exists on the current system.
        Returns:
            (bool): whether or not a GPU is present
        """
        return (
            torch.cuda.is_available()
            or
            torch.backends.mps.is_available()
        )


    def validate_metrics_in_order(self, metrics: list, metric_schemas: list[Schema]) -> list:
        """
        Validates the metrics in the provided list in order to prevent
        emitting exceptions.

        Currently a bit hacky--if the validation fails then an exception
        will be raised and this method will not return. Otherwise if you
        reach the return statement all metrics validated successfully.

        Returns:
            (list): validated metric configuration.
        """
        Schema(
            And(
                list,
                lambda x : len(x) > 0,
                error="Metric list must exist and contain at least one entry."
                )
            ).validate(metrics)

        validated = [
            metric_schemas[i].validate(metrics[i])
            for i in range(len(metrics))
        ]

        return validated


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
        return Schema({
            'optimizer': {
                'name': And(
                    lambda n: self.check_if_optimizer_exists(n),
                    error="Optimizer does not exist. Please ensure the name is correct."
                )
            },
            'metrics': [
                {
                    'name': And(
                        lambda n: self.check_if_metric_exists(n),
                        error="Metric does not exist. Please ensure the name is correct."
                    )
                }
            ]
        }, ignore_extra_keys=True)


    def extract_schema_params(
            self, config_info: dict) -> typing.Optional[dict]:
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

        schema_params['optimizer_schema'] = schema_factory.get_schema_by_name(
            optimizer, 'optimizer', config_info['optimizer']['name']
        )

        schema_params['metric_schemas'] = []

        for metric_info in config_info['metrics']:
            schema_params['metric_schemas'].append(
                schema_factory.get_schema_by_name(
                    metric, 'metric', metric_info['name']
                )
            )

        return schema_params


    def transform_schema(self, config_info: dict) -> dict:
        """
        Transforms the config info passed in by the user to 
        construct the config information required by the model builder.

        Args:
            config_info: dict containing the config information

        Returns:
            (dict): the transformed config info
        """
        config_info['optimizer']['params'] = dict()

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
                'optimizer': schema_params['optimizer_schema'],
                'metrics': Use(
                    lambda x: self.validate_metrics_in_order(x, schema_params['metric_schemas']),
                    error="Specified metric is not valid."
                ),
                'training': {
                    'dataloader': {
                        'batch_size': And(
                            int, schema_utils.is_positive,
                            error="Batch size must be a positive integer."
                        ),
                        'shuffle': And(
                            bool,
                            error="Shuffle must be a boolean value."
                        )
                    },
                    'num_epochs': And(
                        int, schema_utils.is_positive,
                        error="Num_epochs must be a positive integer."
                    )
                },
                'eval': {
                    'dataloader': {
                        'batch_size': And(
                            int, schema_utils.is_positive,
                            error="Batch size must be a positive integer."
                        ),
                        'shuffle': And(
                            bool,
                            error="Shuffle must be a boolean value."
                        )
                    },
                },
                Optional('use_gpu', default=self.check_if_gpu_exists()): Or(
                    False, lambda x: self.check_if_gpu_exists(),
                    error='Cannot set use_gpu to True when no GPU is available.'
                ),
                Optional('run_name', default=None): 
                    Schema(str, error="run_name must be a string"),
                Optional('description', default=None): 
                    Schema(str, error="description must be a string"),
            }
        )

        return config_schema
