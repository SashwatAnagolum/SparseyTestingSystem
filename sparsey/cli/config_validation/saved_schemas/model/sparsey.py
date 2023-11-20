# -*- coding: utf-8 -*-

"""
Sparsey Schema: the schema for Sparsey model config files.
"""


from typing import Optional

from schema import Schema, And

from ..abs_schema import AbstractSchema
from ...saved_schemas import schema_utils

class SparseySchema(AbstractSchema):
    """
    SparseySchema: schema for Sparsey networks.
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

        if (
            ('num_layers' not in config_info) or
            (not isinstance(config_info['num_layers'], int))
        ):
            return None
        
        schema_params['num_layers'] = config_info['num_layers']



        return schema_params


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
        n_layers = schema_params['num_layers']

        config_schema = Schema(
            {
                'input_dimensions': {
                    'width': And(int, schema_utils.is_positive),
                    'height': And(int, schema_utils.is_positive)
                },
                'num_layers': int,
                'layerwise_configs': {
                    'num_macs': And(
                        [And(int, schema_utils.is_positive)],
                        lambda x: schema_utils.is_expected_len(x, n_layers)
                    ),
                    'mac_grid_num_rows': And(
                        [And(int, schema_utils.is_positive)],
                        lambda x: schema_utils.is_expected_len(x, n_layers)
                    ),
                    'mac_grid_num_cols': And(
                        [And(int, schema_utils.is_positive)],
                        lambda x: schema_utils.is_expected_len(x, n_layers)
                    ),
                    'num_cms_per_mac': And(
                        [And(int, schema_utils.is_positive)],
                        lambda x: schema_utils.is_expected_len(x, n_layers)
                    ),
                    'num_neurons_per_cm': And(
                        [And(int, schema_utils.is_positive)],
                        lambda x: schema_utils.is_expected_len(x, n_layers)
                    ),
                    'receptive_field_radii': And(
                        [And(float, schema_utils.is_positive)],
                        lambda x: schema_utils.is_expected_len(x, n_layers)
                    )
                }
            }
        )

        return config_schema
