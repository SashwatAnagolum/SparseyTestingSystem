# -*- coding: utf-8 -*-

"""
Run HPO: script to run HPO
"""


import argparse
import os

# from dotenv import load_dotenv

from sparsepy.cli.config_validation.validate_config import (
    validate_config, get_config_info
)

from sparsepy.tasks.run_hpo import run_hpo


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
        '--training_recipe_config', type=str,
        help='The location of the training recipe config file.'
    )

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    # load_dotenv() # load environment variables from .env

    preprocessing_config_info = get_config_info(
        args.preprocessing_config
    )

    dataset_config_info = get_config_info(
        args.dataset_config
    )

    training_recipe_config_info = get_config_info(
        args.training_recipe_config
    )

    hpo_config_info = get_config_info(
        args.hpo_config
    )

    # preprocessing config validation

    validated_dataset_config = validate_config(
        dataset_config_info, 'dataset', dataset_config_info['dataset_type']
    )

    validated_hpo_config = validate_config(
        hpo_config_info, 'hpo', 'default'
    )

    validated_training_recipe_config = validate_config(
        training_recipe_config_info, 'training_recipe', 'sparsey_hpo'
    )

    run_hpo(
        validated_hpo_config, validated_training_recipe_config,
        validated_dataset_config, preprocessing_config_info,
        # os.getenv("WANDB_API_KEY", "e761ab6db7e51eada8996fa15e9e7eca67414c10")
        "e761ab6db7e51eada8996fa15e9e7eca67414c10"
    )

if __name__ == "__main__":
    main()
