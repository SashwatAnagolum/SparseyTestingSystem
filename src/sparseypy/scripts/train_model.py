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

DESCRIPTION = '''
=====================================
sparseypy: The Sparsey Testing System
=====================================
\n
train_model: train and evaluate custom Sparsey models
\n
--------------------------------------------------------------------------------
\n
Trains and evaluates Sparsey models with user-selected parameters on
user-selected datasets with customizable preprocessing. Automatically logs all
training data to Weights & Biases for later review.
\n
Makes training easy with automatic config file validation and extensible with
custom metrics, datasets, transforms, and more.
\n
Due to the extensive variety of options available, this system uses YAML files
rather than command-line arguments for its configuration.
\n
To use it, you must provide the paths to model, dataset, preprocessing, system,
and training recipe configuration files in the corresponding command-line 
arguments.
\n
For the details of every YAML configuration file and option therein, please see
the commented example configuration files in the "demo" folder in this
project's GitHub repository.
\n
--------------------------------------------------------------------------------
Specifying an existing model: Rather than training a network from scratch
using a path to a model_config file, you can instead specify the name of a 
previously trained model in Weights & Biases as the model_name. 

The system will then download and re-train the chosen model using the provided 
dataset, preprocessing stack, and training recipe.
--------------------------------------------------------------------------------
'''

EPILOG = '''
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
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--training_dataset_config', type=str,
        help='The location of the training dataset config file.'
    )

    parser.add_argument(
        '--evaluation_dataset_config', type=str,
        help='The location of the evaluation dataset config file.'
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
        '--preprocessing_config', type=str,
        help='The location of the preprocessing config file.'
    )

    parser.add_argument(
        '--system_config', type=str,
        help='The location of the system config file.'
    )

    parser.add_argument(
        '--training_recipe_config', type=str,
        help='The location of the trainer config file.'
    )

    args = parser.parse_args()

    return args


def main():
    """
    Main function for the train_model script. Accepts and parses the command line arguments, 
    validates the configuration files with the config schemas, and starts the train_model task.
    """
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

    training_dataset_config_info = get_config_info(
        args.training_dataset_config
    )

    evaluation_dataset_config_info = get_config_info(
        args.evaluation_dataset_config
    )

    if 'console' in system_config_info:
        print_error_stacktrace = system_config_info['console'].get(
            "print_error_stacktrace", False
        )
    else:
        print_error_stacktrace = False

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

    validated_training_dataset_config = validate_config(
        training_dataset_config_info, 'dataset',
        training_dataset_config_info['dataset_type'],
        print_error_stacktrace=print_error_stacktrace
    )

    validated_evaluation_dataset_config = validate_config(
        evaluation_dataset_config_info, 'dataset',
        evaluation_dataset_config_info['dataset_type'],
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
        training_dataset_config=validated_training_dataset_config,
        evaluation_dataset_config=validated_evaluation_dataset_config,
        system_config=validated_system_config
    )

if __name__ == "__main__":
    main()
