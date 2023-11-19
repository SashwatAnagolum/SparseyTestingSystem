# -*- coding: utf-8 -*-

"""
Sparsey Schema: the schema for Sparsey model config files.
"""

from schema import Schema

config_schema = Schema(
    {
        'input_dimensions': {
            'width': int,
            'height': int
        },
        'num_layers': int,
        'layerwise_configs': {
            'num_macs': [int],
            'mac_grid_num_rows': [int],
            'mac_grid_num_cols': [int],
            'num_cms_per_mac': [int],
            'num_neurons_per_cm': [int],
            'receptive_field_radii': [float]
        }
    }
)
