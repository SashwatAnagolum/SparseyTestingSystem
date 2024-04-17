# -*- coding: utf-8 -*-

"""
Train model: script to train models.
"""

import argparse
from argparse import RawDescriptionHelpFormatter

from dotenv import load_dotenv

from sparseypy.cli.config_validation.validate_config import (
    validate_config, get_config_info
)

from sparseypy.tasks.train_model import train_model

description = '''
=====================================
sparseypy: The Sparsey Testing System
=====================================
\n
train_model: train and evaluate Sparsey models
\n
--------------------------------------------------------------------------------
\n\n
To use this script, provide a model configuration, dataset, preprocessint stack,
training recipe, and system configuration. 
\n\n
The system will train and evaluate the model according to the provided 
configuration, logging all results to Weights & Biases and any configured
databases.
\n\n
Due to the extensive variety of parameters available, the Sparsey Testing 
System uses YAML configuration files to provide configuration for all its 
functions.
\n\n
For the details of every YAML configuration file and option therein, please see
the commented example configuration files in the "demo" folder in this
project's GitHub repository.
\n
--------------------------------------------------------------------------------
Specifying a model: As an alternative to providing a network configuration YAML
file as model_config, you can instead specify the name of a previously trained
model in Weights & Biases as the model_name. 

The system will then download and re-train the chosen model using the provided 
dataset, preprocessing stack, and training recipe.
--------------------------------------------------------------------------------
'''

epilogue = '''
--------------------------------------------------------------------------------
Sparsey (c) Dr. Rod Rinkus and Neurithmic Systems. All rights reserved.
--------------------------------------------------------------------------------
'''

def parse_args() -> argparse.Namespace:
    """
    Parses the command line arguments passed in during execution.

    Returns:
        Namespace containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilogue,
        formatter_class=RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--model_config', type=str, required=False,
        help='The location of the model config file. Mutually exclusive with --model_name.'
    )

    parser.add_argument(
        '--model_name', type=str, required=False,
        help='The name of the online model to use. Mutually exclusive with --model_config.'
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

    parser.add_argument(
        '--system_config', type=str,
        help='The location of the system config file.'
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if args.model_config and args.model_name:
        raise ValueError("You can only provide one of --model_config and --model_name.")
    elif not args.model_config and not args.model_name:
        raise ValueError("You must provide either --model_name or --model_config.")

    load_dotenv()

    system_config_info = get_config_info(
        args.system_config
    )

    training_recipe_config_info = get_config_info(
        args.training_recipe_config
    )

    preprocessing_config_info = get_config_info(
        args.preprocessing_config
    )

    dataset_config_info = get_config_info(
        args.dataset_config
    )

    print_error_stacktrace = system_config_info.get("print_error_stacktrace", False)

    validated_system_config = validate_config(
        system_config_info, 'system', 'default',
        print_error_stacktrace=print_error_stacktrace
    )

    validated_trainer_config = validate_config(
        training_recipe_config_info, 'training_recipe', 'sparsey',
        print_error_stacktrace=print_error_stacktrace
    )

    validated_preprocessing_config = validate_config(
        preprocessing_config_info, 'preprocessing_stack', 'default',
        print_error_stacktrace=print_error_stacktrace
    )

    validated_dataset_config = validate_config(
        dataset_config_info, 'dataset', dataset_config_info['dataset_type'],
        print_error_stacktrace=print_error_stacktrace
    )

    if args.model_config:
        model_config_info = get_config_info(
            args.model_config
        )

        model_data = validate_config(
            model_config_info, 'model', 'sparsey',
            print_error_stacktrace=print_error_stacktrace
        )
    else:
        model_data = args.model_name

    train_model(
        model_config=model_data,
        trainer_config=validated_trainer_config,
        preprocessing_config=validated_preprocessing_config,
        dataset_config=validated_dataset_config,
        system_config=validated_system_config
    )

if __name__ == "__main__":
    main()
