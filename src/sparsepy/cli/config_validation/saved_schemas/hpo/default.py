# -*- coding: utf-8 -*-

"""
Default HPO Schema: the schema for HPO runs.
"""


from logging import config
import typing
import sys

from schema import Schema, And, Optional, Or, SchemaError

from sparsepy.cli.config_validation.saved_schemas.abs_schema import AbstractSchema
from sparsepy.cli.config_validation import schema_factory
from sparsepy.cli.config_validation.saved_schemas import (
    model, schema_utils, metric
)


class DefaultHpoSchema(AbstractSchema):
    """
    Default HPO Schema: class for HPO run schemas.
    """
    def extract_schema_params(self, config_info: dict) -> typing.Optional[
        dict
    ]:
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

        return schema_params


    def check_if_model_family_exists(self, model_family) -> bool:
        """
        Checks if a model family with the name model_family exists.

        Returns:
            (bool): whether the model famly exists or not
        """
        try:
            schema_factory.get_schema_by_name(
                model, 'model', model_family
            )
        except ValueError:
            return False

        return True


    def check_if_metric_exists(self, metric_config) -> bool:
        """
        Checks if a metric exists or not.

        Returns:
            (bool): whether the metric exists or not.
        """
        if 'name' not in metric_config:
            return False

        try:
            metric_schema = schema_factory.get_schema_by_name(
                metric, 'metric', metric_config['name']
            )
        except ValueError:
            return False

        try:
            metric_schema.validate(metric_config)
        except SchemaError as e:
            print(e)

            return False

        return True


    def check_optimizer_hyperparams_validity(self, config_info):
        """
        Checks whether the config for the hyperparameters to be
        optimized is valid or not.

        Returns:
            (bool): whether the config is valid or not.
        """
        hyperparam_schema = Schema(
            Or(
                {
                    'min': Or(int, float),
                    'max': Or(int, float),
                    'distribution': Or(
                        'int_uniform', 'uniform', 'categorical',
                        'q_uniform', 'log_uniform', 'log_uniform_values',
                        'q_log_uniform', 'q_log_uniform_values',
                        'inv_log_uniform', 'normal', 'q_normal',
                        'log_normal', 'q_log_normal'
                    )
                },
                {
                    'values': And(
                        list,
                        schema_utils.all_elements_are_same_type
                    )
                },
                {
                    'value': Or(str, int, float, bool)
                }
            )
        )

        if isinstance(config_info, dict):
            if (
                'min' in config_info.keys() or
                'values' in config_info.keys() or
                'value' in config_info.keys()
            ):
                try:
                    hyperparam_schema.validate(config_info)
                except SchemaError as e:
                    print(e)
                    sys.exit(0)
            else:
                for key, value in config_info.items():
                    self.check_optimizer_hyperparams_validity(value)
        elif isinstance(config_info, list):
            for config_item in config_info:
                self.check_optimizer_hyperparams_validity(config_item)
        else:
            raise ValueError(
                f'{config_info} is not a valid configuration' +
                'for hyperparameters to optimize!'
            )

        return True


    def transform_schema(self, config_info: dict) -> dict:
        """
        Transforms the config info passed in by the user to 
        construct the config information required to build the HPORun.

        Args:
            config_info: dict containing the config information

        Returns:
            dict containing the transformed config info
        """
        for i in range(
            len(config_info['optimization_objective']['objective_terms'])
        ):
            config_info['optimization_objective'][
                'objective_terms'
            ][i]['metric'] = schema_factory.get_schema_by_name(
                metric, 'metric',
                config_info['optimization_objective']
                ['objective_terms'][i]['metric']['name']
            ).validate(
                config_info['optimization_objective'][
                    'objective_terms'
                ][i]['metric']
            )

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
                'model_family': And(str, self.check_if_model_family_exists),
                'hpo_run_name': str,
                'project_name': str,
                'hyperparameters': And(
                    dict, self.check_optimizer_hyperparams_validity
                ),
                'hpo_strategy': Or('random', 'grid', 'bayes'),
                'optimization_objective': {
                    'objective_terms': [
                        {
                            'metric': self.check_if_metric_exists,
                            'weight': float,
                        }
                    ],
                    'combination_method': 'sum'
                },
                'num_candidates': And(int, schema_utils.is_positive)
            }
        )

        return config_schema
