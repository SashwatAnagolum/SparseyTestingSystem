# -*- coding: utf-8 -*-

"""
HPO Run: file holding the HPORun class.
"""


import random
import wandb

from sparsepy.core.metrics.metric_factory import MetricFactory
from sparsepy.access_objects.models.model_builder import ModelBuilder
from sparsepy.cli.config_validation.validate_config import validate_config

class HPORun():
    """
    HPORun: class for performing HPO Runs.

    Attributes:
        num_steps_to_perform (int): the total number of 
            candidates to try out during the HPO process
    """
    def __init__(self, hpo_config: dict, trainer_config: dict,
        dataset_config: dict, preprocessing_config: dict,
        wandb_api_key: str):
        """
        Initializes the HPORun object.

        Args:
            hpo_config (dict): configurations for the HPO Run.
            trainer_config (dict): configruations for the
                training recipe.
            dataset_config (dict): configurations for the dataset.
            preprocessing_config (dict): configurations for the
                preprocessing stack.
            wandb_api_key (str): the Weights and Biases API key to 
                use to login to WandB and log data.
        """
        wandb.login(key=wandb_api_key, verify=True)

        self.sweep_config = self.construct_sweep_config(hpo_config)
        self.sweep_id = wandb.sweep(sweep=self.sweep_config)
        self.num_trials = hpo_config['num_candidates']
        self.config_info = hpo_config


    def check_is_value_constraint(self, config):
        """
        Checks if a piece of config is a constraint set for 
            a hyperparameter.

        Args:
            config (dict): the piece of config to check.

        Returns:
            (bool): whether config is a constraint set for
                a hyperparameter or not.
        """
        if not isinstance(config, dict):
            return False

        keys = set(config.keys())

        if len(keys) == 1:
            return ('value' in keys) or ('values' in keys)
        elif len(keys) == 3:
            return keys == {'min', 'max', 'distribution'}

        return False


    def extract_hyperparams(self, hyperparams_config: dict) -> dict:
        """
        Extract the hyperparameters for a WandB sweep.

        Args:
            hyperparams_config (dict): config information for the
                network hyperparameters
        
        Returns:
            (dict): parameter information required for the WandB sweep
                construction.
        """
        sweep_parameters = dict()

        for key, value in hyperparams_config.items():
            if isinstance(value, dict):
                if self.check_is_value_constraint(value):
                    sweep_parameters[key] = value
                else:
                    sweep_parameters[key] = dict()
                    sweep_parameters[key][
                        'parameters'
                    ] = self.extract_hyperparams(value)
            elif isinstance(value, list):
                for index, list_element in enumerate(value):
                    element_key = f'{key}_{index}'

                    if self.check_is_value_constraint(list_element):
                        sweep_parameters[element_key] = list_element
                    else:
                        sweep_parameters[element_key] = dict()
                        sweep_parameters[element_key][
                            'parameters'
                        ] = self.extract_hyperparams(list_element)                    

        return sweep_parameters


    def construct_sweep_config(self, hpo_config: dict) -> dict:
        """
        Construct the sweep configuration for the Weights and Biases
        sweep to be performed as part of the HPO run.

        Args:
            hpo_config (dict): configuration info for the HPO run.

        Returns:
            (dict): the WandB sweep configuration.
        """
        sweep_hyperparams = self.extract_hyperparams(
            hpo_config['hyperparameters']
        )

        sweep_config = {
            'method': hpo_config['hpo_strategy'],
            'name': hpo_config['hpo_run_name'],
            'metric': {'goal': 'minimize', 'name': 'hpo_objective'},
            'project': hpo_config['project_name'],
            'run_cap': hpo_config['num_candidates'],
            'parameters': sweep_hyperparams
        }

        return sweep_config


    def generate_model_config(self, wandb_config: dict) -> dict:
        model_config = dict()
        layer_keys = dict()
        layers = []

        for key, value in wandb_config.items():
            if ('layers_' in key):
                layer_keys[key] = value
            elif (key != 'num_layers'):
                model_config[key] = value

        for i in range(len(layer_keys)):
            layers.append(layer_keys[f'layers_{i}'])

        model_config['layers'] = layers

        return model_config


    def step(self) -> None:
        """
        Perform one HPO step, which includes sampling 
        a set of model hyperparameters, training the created
        model, and computing the user-specified objective function
        using the trained model.
        """
        wandb.init()
        
        model_config = self.generate_model_config(
            dict(wandb.config)
        )

        validated_config, _ = validate_config(
            model_config, 'model', self.config_info['model_family']
        )

        if validated_config is not None:
            model = ModelBuilder.build_model(validated_config)

        wandb.log({'hpo_objective': random.random()})


    def run_sweep(self) -> dict:
        """
        Run the HPO process based on the WandB sweep config created.
        """
        wandb.agent(
            self.sweep_id, self.step,
            count=self.num_trials
        )

        wandb._teardown()

        return dict()
