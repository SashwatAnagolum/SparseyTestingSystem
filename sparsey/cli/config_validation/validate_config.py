# -*- coding: utf-8 -*-

"""
Validate Config: functions to validate a config file
    provided by a user against a provided schema.
"""


import os
import argparse
import sys
import yaml

from .schemas import get_schema_by_name
from .saved_schemas.abs_schema import AbstractSchema


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
    config_schema = get_schema_by_name(schema_type, schema_name)

    return config_schema


def validate_config(config_filepath: str, schema_type: str,
    schema_name: str) -> dict:
    """
    Validates the given config file against the given schema. If
    the config is valid, then the valid config is returned, otherwise
    an error is thrown.

    Args:
        config_filepath: a str containing the filepath to the config file.
        schema_type: a str containing the type of schema
            (model / trainer / HPO / plot).
        schema_name: a str containing the name of the schema to validate
            the config file against.

    Returns:
        A bool indicating whether the config file is valid or not.
    """
    config_info = get_config_info(config_filepath)
    config_schema_class = get_schema(schema_type, schema_name)

    schema_obj = config_schema_class()
    valid_config = schema_obj.validate(config_info)

    if valid_config is None:
        raise ValueError(
            'The provided config file does not match the required schema!\n'
        )

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

    valid_config = validate_config(args.config_filepath, args.schema_type, args.schema_name)
    
    print(valid_config)

if __name__ == "__main__":
    main()
