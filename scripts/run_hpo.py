# -*- coding: utf-8 -*-

"""
Run HPO: script to run HPO
"""


import argparse

from sparsepy.cli.config_validation.validate_config import (
    validate_config, get_config_info
)


def parse_args() -> argparse.Namespace:
    """
    Parses the command line arguments passed in during execution.

    Returns:
        Namespace containing the parsed arguments.
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--training_recipe_config', type=str,
        help='The location of the trainer config file.'
    )
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

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    
    training_recipe_config_info = get_config_info(
        args.training_recipe_config
    )

    validated_trainer_config, is_valid = validate_config(
        training_recipe_config_info, 'training_recipe', 'sparsey'
    )
    
    preprocessing_config_info = get_config_info(
        args.preprocessing_config
    )

    # preprocessing config validation

    dataset_config_info = get_config_info(
        args.dataset_config
    )
    validated_dataset_config, _ = validate_config(
        dataset_config_info, 'dataset', 'image'
    )

    hpo_config_info = get_config_info(
        args.hpo_config
    )

    # TODO update to validate the config
    validated_hpo_config = validate_config(
        hpo_config_info, 'hpo', 'default'
    )

    # TODO start the HPO run

if __name__ == "__main__":
    main()
