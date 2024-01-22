# -*- coding: utf-8 -*-

"""
Default HPO Schema: the schema for HPO runs.
"""


import typing
import sys

from schema import Schema, And, Optional, Or, SchemaError

from sparsepy.cli.config_validation.saved_schemas.abs_schema import AbstractSchema
from sparsepy.cli.config_validation import schema_factory
from sparsepy.cli.config_validation.saved_schemas import (
    model, hpo_strategy, schema_utils
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

        try:
            hpo_strategy_name = config_info['hpo_strategy']['name']
        except KeyError as e:
            raise ValueError('Missing HPO Strategy name!') from e
        try:
            hpo_strategy_schema = schema_factory.get_schema_by_name(
                hpo_strategy, 'hpo_strategy', hpo_strategy_name
            )

            schema_params['hpo_strategy_schema'] = hpo_strategy_schema
        except ValueError as e:
            raise ValueError(
                f'Invalid HPO Strategy name {hpo_strategy_name}!'
            ) from e
    
        if 'optimization_objective' not in config_info:
            raise ValueError('`optimization_objective` config missing!')
        elif not isinstance(config_info['optimization_objective'], list):
            raise ValueError(
                'invalid `optimization_objective` config' + 
                f'{config_info["optimization_objective"]}: must be a list!'
            )      

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
                    'bounds': And(
                        list,
                        lambda x: schema_utils.is_expected_len(x, 2)
                    ),
                    'data_type': lambda x: x in ['int', 'float']
                },
                {
                    'value_set': And(
                        list,
                        lambda x: sum(
                            [
                                True if isinstance(i, int)
                                or isinstance(i, float)
                                else False for i in x
                            ]
                        ) == len(x)
                    )
                },
            )
        )

        if isinstance(config_info, dict):
            if 'bounds' in config_info.keys() or 'value_set' in config_info.keys():
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
                Optional('fixed_hyperparameters'): dict,
                'optimized_hyperparameters': And(
                    dict, self.check_optimizer_hyperparams_validity
                ),
                'hpo_strategy': {
                    'name': str,
                    Optional(
                        'params',
                        default=None
                    ): schema_params['hpo_strategy_schema']
                },
                'optimization_objective': [
                    {
                        'name': str,
                        Optional('params', default=None): dict,
                        'weight': float
                    }
                ]
            }
        )

        return config_schema
