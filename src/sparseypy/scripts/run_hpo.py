# -*- coding: utf-8 -*-

"""
Run HPO: script to run HPO
"""

import argparse
from argparse import RawDescriptionHelpFormatter

from dotenv import load_dotenv

from sparseypy.cli.config_validation.validate_config import (
    validate_config, get_config_info
)

from sparseypy.tasks.run_hpo import run_hpo

DESCRIPTION = '''
=====================================
sparseypy: The Sparsey Testing System
=====================================
\n
run_hpo: automatic hyperparameter optimization for Sparsey models
\n
--------------------------------------------------------------------------------
\n
Performs hyperparameter optimization using a user-selected set of
hyperparameters. 
\n
Allows customization of all model- and training-related hyperparameters using a
flexible set of value options and multiple HPO strategies with intelligent
config file validation and automatic logging of data to Weights & Biases and
Firestore.
\n
Due to the extensive variety of options available, this system uses YAML files
rather than command-line arguments for its configuration.
\n
To use it, you must provide the paths to HPO, dataset, preprocessing, 
and system configuration files in the corresponding command-line arguments.
\n
For the details of every YAML configuration file and option therein, please see
the commented example configuration files in the "demo" folder in this
project's GitHub repository.
\n
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
        '--hpo_config', type=str,
        help='The location of the hyperparameter optimization (HPO) config file.'
    )

    parser.add_argument(
        '--preprocessing_config', type=str,
        help='The location of the preprocessing config file.'
    )

    parser.add_argument(
        '--system_config', type=str,
        help='The location of the system config file.'
    )

    args = parser.parse_args()

    return args

def main():
    """
    Main function for the run_hpo script. Accepts and parses the command line arguments, 
    validates the configuration files with the config schemas, and starts the run_hpo task.
    """
    args = parse_args()

    load_dotenv() # load environment variables from .env

    system_config_info = get_config_info(
        args.system_config
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

    hpo_config_info = get_config_info(
        args.hpo_config
    )

    print_error_stacktrace = system_config_info['console'].get('print_error_stacktrace', False)

    validated_preprocessing_config = validate_config(
        preprocessing_config_info, 'preprocessing_stack', 'default',
        print_error_stacktrace=print_error_stacktrace
    )

    validated_system_config = validate_config(
        system_config_info, 'system', 'default',
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

    validated_hpo_config = validate_config(
        hpo_config_info, 'hpo', 'default',
        print_error_stacktrace=print_error_stacktrace
    )

    run_hpo(
        hpo_config=validated_hpo_config,
        training_dataset_config=validated_training_dataset_config,
        evaluation_dataset_config=validated_evaluation_dataset_config,
        preprocessing_config=validated_preprocessing_config,
        system_config=validated_system_config
    )

if __name__ == "__main__":
    main()
