# -*- coding: utf-8 -*-

"""
Train model: script to train models.
"""

import argparse

from sparsepy.cli.config_validation.validate_config import (
    validate_config, get_config_info
)

from sparsepy.tasks.train_model import train_model


def parse_args() -> argparse.Namespace:
    """
    Parses the command line arguments passed in during execution.

    Returns:
        Namespace containing the parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_config', type=str,
        help='The location of the model config file.'
    )

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

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    model_config_info = get_config_info(args.model_config)
    validated_config = validate_config(
        model_config_info, 'model', 'sparsey'
    )

    training_recipe_config_info = get_config_info(
        args.training_recipe_config
    )

    validated_trainer_config = validate_config(
        training_recipe_config_info, 'training_recipe', 'sparsey'
    )

    preprocessing_config_info = get_config_info(
        args.preprocessing_config
    )

    validated_preprocessing_config = validate_config(
        preprocessing_config_info, 'preprocessing_stack', 'default'
    )
    
    dataset_config_info = get_config_info(
        args.dataset_config
    )

    validated_dataset_config = validate_config( # needs updating to support different dataset types
        dataset_config_info, 'dataset', dataset_config_info['dataset_type']
    )

    train_model(
        validated_config,
        validated_trainer_config,
        preprocessing_config_info,
        validated_dataset_config
    )

if __name__ == "__main__":
    main()