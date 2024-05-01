# -*- coding: utf-8 -*-

"""
Sparsey Model Schema: the schema for Sparsey model config files.
"""


import typing
import math

from schema import Schema, And, Optional, Or, Use

from sparseypy.cli.config_validation.saved_schemas.abs_schema import AbstractSchema
from sparseypy.cli.config_validation.saved_schemas import schema_utils
from sparseypy.core import hooks


class SparseyModelSchema(AbstractSchema):
    """
    SparseyModelSchema: schema for Sparsey networks.
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


    def check_if_hook_exists(self, hook_name):
        """
        Checks if a hook exists.

        Args: 
            hook_name (str): name of the hook

        Returns:
            (bool): whether the hook exists in the system or not.
        """
        hook_name = ''.join(
            [i.capitalize() for i in hook_name.split('_')] + ['Hook']
        )

        return hook_name in dir(hooks)


    def compute_factor_pair(self, num: int) -> typing.Tuple[int, int]:
        """
        Returns the pair of factors whose product is num
        whose elements are closest to sqrt(num)

        Args:
            num (int): the number to find the factors of.

        Returns:
            (Tuple[int, int]): the chosen factors
        """
        factor = 1

        for i in range(2, math.floor(num ** 0.5) + 1):
            if not num % i:
                factor = i

        return factor, num // factor


    def compute_grid_size(self, num_macs: int) -> typing.Tuple[int, int]:
        """
        Finds the smallest grid with at least 2 rows 
        that can accomodate num_macs.
        """
        factor_1 = 1

        while factor_1 == 1:
            factor_1, factor_2 = self.compute_factor_pair(num_macs)
            num_macs += 1

        return factor_1, factor_2


    def transform_schema(self, config_info: dict) -> dict:
        """
        Transforms the config info passed in by the user to 
        construct the config information required by the model builder.

        Args:
            config_info: dict containing the config information

        Returns:
            dict containing the transformed config info
        """
        prev_layer_dims = (
            config_info['input_dimensions']['width'],
            config_info['input_dimensions']['height'],
            config_info['input_dimensions']['width'] *
            config_info['input_dimensions']['height'],
            1, 1, 'rect'
        )

        for index in range(len(config_info['layers'])):
            config_info['layers'][index]['params'][
                'prev_layer_mac_grid_num_rows'
            ] = prev_layer_dims[0]

            config_info['layers'][index]['params'][
                'prev_layer_mac_grid_num_cols'
            ] = prev_layer_dims[1]

            config_info['layers'][index]['params'][
                'prev_layer_num_macs'
            ] = prev_layer_dims[2]

            config_info['layers'][index]['params'][
                'prev_layer_num_cms_per_mac'
            ] = prev_layer_dims[3]

            config_info['layers'][index]['params'][
                'prev_layer_num_neurons_per_cm'
            ] = prev_layer_dims[4]

            config_info['layers'][index]['params'][
                'prev_layer_grid_layout'
            ] = prev_layer_dims[5]

            if config_info['layers'][index]['params']['autosize_grid']:
                num_rows, num_cols = self.compute_factor_pair(
                    config_info['layers'][index]['params']['num_macs']
                )

                config_info['layers'][index]['params']['mac_grid_num_rows'] = num_rows
                config_info['layers'][index]['params']['mac_grid_num_cols'] = num_cols

            config_info['layers'][index]['params'][
                'activation_threshold_min'
            ] = float(
                config_info['layers'][index]['params'][
                    'activation_threshold_min'
                ]
            )

            config_info['layers'][index]['params'][
                'activation_threshold_max'
            ] = float(
                config_info['layers'][index]['params'][
                    'activation_threshold_max'
                ]
            )

            prev_layer_dims = (
                config_info['layers'][index]['params']['mac_grid_num_rows'],
                config_info['layers'][index]['params']['mac_grid_num_cols'],
                config_info['layers'][index]['params']['num_macs'],
                config_info['layers'][index]['params']['num_cms_per_mac'],
                config_info['layers'][index]['params']['num_neurons_per_cm'],
                config_info['layers'][index]['params']['grid_layout'],
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
                Optional('model_name', default=None): And(str, error="Model name must be a string"),
                Optional('model_description', default=None): And(str, error="Model description must be a string"),
                'input_dimensions': {
                    'width': And(int, schema_utils.is_positive, error="Width must be a positive integer"),
                    'height': And(int, schema_utils.is_positive, error="Height must be a positive integer")
            },
            'layers': [
                {
                    'name': And(str, lambda n: n == 'sparsey', error="Layer name must be 'sparsey'"),
                    'params': {
                        Optional('autosize_grid', default=False): bool,
                        Optional('grid_layout', default='rect'): Or('rect', 'hex', error="Grid layout must be 'rect' or 'hex'"),
                        'num_macs': And(int, schema_utils.is_positive, error="Number of MACs must be a positive integer"),
                        Optional('mac_grid_num_rows', default=1): And(int, schema_utils.is_positive, error="MAC grid number of rows must be a positive integer"),
                        Optional('mac_grid_num_cols', default=1): And(int, schema_utils.is_positive, error="MAC grid number of columns must be a positive integer"),
                        'num_cms_per_mac': And(int, schema_utils.is_positive, error="Number of CMs per MAC must be a positive integer"),
                        'num_neurons_per_cm': And(int, schema_utils.is_positive, error="Number of neurons per CM must be a positive integer"),
                        'mac_receptive_field_size': And(Or(Use(float)), schema_utils.is_positive, error="MAC receptive field size must be a positive number"),
                        'sigmoid_lambda': And(Or(Use(float), int), schema_utils.is_positive, error="Sigmoid lambda must be a positive number"),
                        'sigmoid_phi': Or(Use(float), error="Sigmoid phi must be an integer or float"),
                        'saturation_threshold': And(float, lambda n: 0 <= n <= 1, error="Saturation threshold must be between 0 and 1"),
                        'activation_threshold_min': And(Or(Use(float)), lambda x: schema_utils.is_between(x, 0.0, 1.0), error="Activation threshold min must be between 0 and 1"),
                        'activation_threshold_max': And(Or(Use(float)), lambda x: schema_utils.is_between(x, 0.0, 1.0), error="Activation threshold max must be between 0 and 1"),
                        'sigmoid_chi': Or(Use(float), error="Sigmoid chi must be an integer or float"),
                        'min_familiarity': And(float, lambda x: 0 <= x < 1, error="Min familiarity must be between 0 and 1"),
                        'permanence_steps': And(int, Use(float), error='num_steps_to_zero'),
                        'permanence_convexity': And(
                            Use(float),
                            lambda n: 0 < n,
                            error='convexity must be a float > 0'
                        )
                    }
                }
            ],
            Optional('hooks'): [
                {
                    'name': And(str, self.check_if_hook_exists, error="Specified hook does not exist")
                }
            ]
        })

        return config_schema
