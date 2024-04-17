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

from sparseypy.tasks.evaluate_model import evaluate_model

description = '''
=====================================
sparseypy: The Sparsey Testing System
=====================================
\n
evaluate_model: evaluate previously-trained models on additional datasets
\n
--------------------------------------------------------------------------------
\n\n
To use this script, specify the name of an existing model in Weights & Biases 
and provide a dataset, a preprocessing stack, and a training configuration. 
\n\n
The system will automatically retrieve the model and perform evaluation on the 
indicated dataset, logging all results to Weights & Biases.
\n\n
Due to the extensive variety of parameters available, the Sparsey Testing 
System uses YAML configuration files to provide the bulk of its configuration
and the 
\n\n
For the details of every YAML configuration file and option therein, please see
the commented example configuration files in the "demo" folder in this
project's GitHub repository.
\n
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
        '--model_name', type=str,
        help='The name of the online model to evaluate.'
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

    evaluate_model(
        model_name=args.model_name,
        trainer_config=validated_trainer_config,
        preprocessing_config=validated_preprocessing_config,
        dataset_config=validated_dataset_config,
        system_config=validated_system_config
    )

if __name__ == "__main__":
    main()