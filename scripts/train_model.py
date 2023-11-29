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
        '--trainer_config', type=str,
        help='The location of the trainer config file.'
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    config_info = get_config_info(args.model_config)
    validated_config, is_valid = validate_config(
        config_info, 'model', 'sparsey'
    )

    trainer_config_info = get_config_info(args.trainer_config)
    validated_trainer_config, is_valid = validate_config(
        trainer_config_info, 'trainer', 'sparsey'
    )

    train_model(validated_config, validated_trainer_config, None, None)

if __name__ == "__main__":
    main()