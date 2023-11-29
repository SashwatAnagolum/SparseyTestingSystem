# -*- coding: utf-8 -*-

"""
Sparsey Model Schema: the schema for Sparsey model config files.
"""


from typing import Optional

from schema import Schema, And

from ..abs_schema import AbstractSchema
from ...saved_schemas import schema_utils


class SparseyModelSchema(AbstractSchema):
    """
    SparseyModelSchema: schema for Sparsey networks.
    """
    def extract_schema_params(self, config_info: dict) -> Optional[dict]:
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
            1, 1
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

            prev_layer_dims = (
                config_info['layers'][index]['params']['mac_grid_num_rows'],
                config_info['layers'][index]['params']['mac_grid_num_rows'],
                config_info['layers'][index]['params']['num_macs'],
                config_info['layers'][index]['params']['num_cms_per_mac'],
                config_info['layers'][index]['params']['num_neurons_per_cm']
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
                'input_dimensions': {
                    'width': And(int, schema_utils.is_positive),
                    'height': And(int, schema_utils.is_positive)
                },
                'layers': [
                    {
                        'name': And(str, 'sparsey'),
                        'params': {
                            'num_macs': And(int, schema_utils.is_positive),
                            'mac_grid_num_rows': And(int, schema_utils.is_positive),
                            'mac_grid_num_cols': And(int, schema_utils.is_positive),
                            'num_cms_per_mac': And(int, schema_utils.is_positive),
                            'num_neurons_per_cm': And(int, schema_utils.is_positive),
                            'mac_receptive_field_radius': And(
                                float,
                                schema_utils.is_positive,
                                lambda x: schema_utils.is_between(x, 0.0, 1.0)
                            )
                        }
                    }
                ]
            }
        )

        return config_schema
