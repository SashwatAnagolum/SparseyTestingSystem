# -*- coding: utf-8 -*-

"""
Run HPO: script to run HPO
"""


import argparse
import os

from dotenv import load_dotenv

from sparseypy.cli.config_validation.validate_config import (
    validate_config, get_config_info
)

from sparseypy.tasks.run_hpo import run_hpo


def parse_args() -> argparse.Namespace:
    """
    Parses the command line arguments passed in during execution.

    Returns:
        Namespace containing the parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--preprocessing_config', type=str,
        help='The location of the preprocessing config file.'
    )

    parser.add_argument(
        '--dataset_config', type=str,
        help='The location of the dataset config file.'
    )

    parser.add_argument(
        '--hpo_config', type=str,
        help='The location of the hyperparameter optimization (HPO) config file.'
    )


    parser.add_argument(
        '--system_config', type=str,
        help='The location of the system config file.'
    )

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    load_dotenv() # load environment variables from .env

    system_config_info = get_config_info(
        args.system_config
    )

    preprocessing_config_info = get_config_info(
        args.preprocessing_config
    )

    dataset_config_info = get_config_info(
        args.dataset_config
    )


    hpo_config_info = get_config_info(
        args.hpo_config
    )

    # preprocessing config validation

    validated_system_config = validate_config(
        system_config_info, 'system', 'default'
    )

    validated_dataset_config = validate_config(
        dataset_config_info, 'dataset', dataset_config_info['dataset_type']
    )

    validated_hpo_config = validate_config(
        hpo_config_info, 'hpo', 'default'
    )



    run_hpo(
        validated_hpo_config,
        validated_dataset_config, preprocessing_config_info,
        validated_system_config
    )

if __name__ == "__main__":
    main()
