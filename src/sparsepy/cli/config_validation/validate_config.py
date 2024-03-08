# -*- coding: utf-8 -*-

"""
Validate Config: functions to validate a config file
    provided by a user against a provided schema.
"""


import os
import argparse
import sys
from typing import Tuple, Optional

import yaml

from .schema_factory import get_schema_by_name
from .saved_schemas.abs_schema import AbstractSchema
from . import saved_schemas

def get_config_info(config_filepath: str) -> dict:
    """
    Reads the file at the filepath passed in by the user.

    Args:
        config_filepath: a str containing a fileapath to the config file.

    Returns:
        a dict containing the parsed information in the file located
            at config_filepath.
    """
    if not os.path.exists(config_filepath):
        raise IOError(f'No file exists at location {config_filepath}!')

    config_file = open(config_filepath, 'r', encoding='utf-8')

    try:
        config_info = yaml.safe_load(config_file)
    except Exception as e:
        print(f'Error while reading config file: {e}')
        sys.exit(-1)

    return config_info


def get_schema(schema_type, schema_name) -> AbstractSchema:
    """
    Gets and returns the schema corresponding to the passed
    in schema type and name.

    Args:
        schema_type: a str containing the type of schema
            (model / trainer / HPO / plot).
        schema_name: a str containing the name of the schema to validate
            the config file against.

    Returns:
        the corresponding AbstractSchema
    """
    try:
        schema_module = getattr(saved_schemas, schema_type)
    except Exception:
        raise ValueError(f'Invalid schema type {schema_type}!')

    config_schema = get_schema_by_name(
        schema_module, schema_type, schema_name
    )

    return config_schema


def validate_config(config_info: dict, schema_type: str,
    schema_name: str) -> dict:
    """
    Validates the given config file against the given schema. If
    the config is valid, then (the valid config, True) is returned,
    otherwise a SchemaError is thrown.

    Args:
        config_info: a dict containing the config info passed in.
        schema_type: a str containing the type of schema
            (model / trainer / HPO / plot).
        schema_name: a str containing the name of the schema to validate
            the config file against.

    Returns:
        (dict) the valid config info.
    """
    config_schema = get_schema(schema_type, schema_name)
    valid_config = config_schema.validate(config_info)

    return valid_config


def main() -> None:
    """
    Reads the passed in command line arguments before performing
    config file validation.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filepath')
    parser.add_argument('--schema_type')
    parser.add_argument('--schema_name')

    args = parser.parse_args()

    config_info = get_config_info(args.config_filepath)
    valid_config = validate_config(
        config_info, args.schema_type, args.schema_name
    )

if __name__ == "__main__":
    main()
