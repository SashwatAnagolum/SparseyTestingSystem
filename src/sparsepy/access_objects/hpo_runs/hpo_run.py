# -*- coding: utf-8 -*-

"""
HPO Run: file holding the HPORun class.
"""

import torch
import random
import wandb
import traceback

from copy import deepcopy

from sparsepy.core.metrics.metric_factory import MetricFactory
from sparsepy.core.results import HPOResult, HPOStepResult
from sparsepy.core.data_storage_retrieval import DataStorer
from sparsepy.access_objects.models.model_builder import ModelBuilder
from sparsepy.cli.config_validation.validate_config import validate_config
from sparsepy.core.hpo_objectives.hpo_objective import HPOObjective
from sparsepy.access_objects.training_recipes.training_recipe_builder import (
    TrainingRecipeBuilder
)


class HPORun():
    """
    HPORun: class for performing HPO Runs.

    Attributes:
        num_steps_to_perform (int): the total number of 
            candidates to try out during the HPO process
    """
    def __init__(self, hpo_config: dict, trainer_config: dict,
        dataset_config: dict, preprocessing_config: dict):
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

        self.sweep_config = self.construct_sweep_config(hpo_config)
        self.sweep_id = wandb.sweep(sweep=self.sweep_config)
        self.num_trials = hpo_config['num_candidates']
        self.config_info = hpo_config

        trainer_config['metrics'] = hpo_config['metrics']

        self.preprocessing_config = preprocessing_config
        self.dataset_config = dataset_config
        self.training_recipe_config = trainer_config

        
        # BUG does this approach log things in an incorrect order for multithreaded runs?
        logged_configs = {
            'hpo_config': hpo_config,
            'sweep_config': self.sweep_config, # do we need to log this?
            'dataset_config': dataset_config,
            'training_recipe_config': trainer_config,
            'preprocessing_config': preprocessing_config
        }
        
        # create the DataStorer
        self.data_storer = DataStorer(trainer_config['metrics'])

        # create the HPOResult (also sets start time)
        self.hpo_results = HPOResult(logged_configs, self.sweep_id, hpo_config['hpo_run_name'])

        # create the sweep
        self.data_storer.create_hpo_sweep(self.hpo_results)

        # only initialize the objective once, in the constructor
        self.objective = HPOObjective(hpo_config)
        self.best = None
        self.best_results = None
        self.best_run = 0
        self.num_steps = 0


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

        for i in range(wandb_config['num_layers']):
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

        validated_config = validate_config(
            model_config, 'model', self.config_info['model_family'], survive_with_exception=True
        )

        try:
            if validated_config is not None:
                model = ModelBuilder.build_model(validated_config)

                training_recipe = TrainingRecipeBuilder.build_training_recipe(
                    model, deepcopy(self.dataset_config),
                    deepcopy(self.preprocessing_config),
                    deepcopy(self.training_recipe_config)
                )

                done = False
                results = None

                # do we need to move this earlier?
                hpo_step_results = HPOStepResult(parent_run=self.sweep_id, id=wandb.run.id, configs={
                    'dataset_config': self.dataset_config,
                    'preprocessing_config': self.preprocessing_config,
                    'training_recipe_config': self.training_recipe_config,
                    'model_config': validated_config
                })

                # increment step counter
                self.num_steps += 1

                # perform training
                while not done:
                    # are we supposed to only use the results from the last step in objective computation?
                    # we might need to change this
                    results, done = training_recipe.step()
                # fetch training results
                training_results = training_recipe.get_summary("training")
                # perform evaluation
                done = False
                while not done:
                    results, done = training_recipe.step(training=False)
                # fetch evaluation results
                eval_results = training_recipe.get_summary("evaluation")

                # calculate the objective from the evaluation results
                objective_results = self.objective.combine_metrics(eval_results)
                
                if results is not None:
                    # final result ready

                    # populate the HPOStepResult
                    hpo_step_results.populate(objective=objective_results, 
                                              training_results=training_results,
                                              eval_results=eval_results)

                    # add the HPOStepResults to the HPOResult
                    self.hpo_results.add_step(hpo_step_results)

                    # OLD LOGIC
                    # this one is the best one if 1) there is no previous result or 2) its objective value is higher than the previous best result
                    is_best = (not self.best) or (self.best and objective_results["total"] > self.best.get_objective()["total"])

                    print(f"Completed trial {self.num_steps} of {self.num_trials}")
                    if self.best:
                        print(f"Previous best objective value: {self.best.get_objective()['total']:.5f}")

                    self._print_breakdown(hpo_step_results)

                    if is_best:
                        self.best = hpo_step_results
                        self.best_results = results
                        self.best_run = self.num_steps
                        self.best_config = validated_config
                        print(f"This is the new best value!")

                    self.data_storer.save_hpo_step(wandb.run.sweep_id, hpo_step_results)

                    # cache run path for updating config
                    run_path = wandb.run.path
                    # finish the run - wandb.run may no longer be correct below this point
                    wandb.finish()
                    # strip unused layers from W&B side config
                    # this must occur after .finish() due to a bug in W&B
                    max_layers = len(model_config['layers'])
                    run = wandb.Api().run(run_path)
                    for k in run.config.copy():
                       if ("layers_" in k) and (int(k[7:]) >= max_layers):
                           del run.config[k]
                    run.update()
                    #return HPOResult(results, objective_results)

                    # if this is the final run, also log the best-performing model
                    # this should be handled by the DS when it is called to store the HPORun from run_sweep()
                    #if self.num_steps >= self.num_trials:
                    #    print(f"OPTIMIZATION RUN COMPLETED")
                    #    print(f"Best run: {self.best_run}")
                    #    self._print_breakdown(self.best)
                    #    print(f"Best run configuration: {self.best_config}")

                        #wandb.log(
                        #    {
                        #        'best_objective': self.best["total"],
                        #        'best_objective_breakdown': self.best,
                        #        'best_results': self.best_results,
                        #        'best_params': self.best_config
                        #    }
                        #)
        except Exception as e:
            # log HPOStep failure? otherwise order of items in HPOResult will not match total number of steps/step order
            print(traceback.format_exc())


    def _print_breakdown(self, step_results: HPOStepResult):
        objective_results = step_results.get_objective()
        # enhance with summary of metrics
        print(f"Objective value: {objective_results['total']:.5f}")
        print(f"Combination method: {objective_results['combination_method']}")
        print("Objective term breakdown:")
        for name, values in objective_results["terms"].items():
            print(f"* {name:>25}: {values['value']:.5f} with weight {values['weight']}")


    def run_sweep(self) -> HPOResult:
        """
        Run the HPO process based on the WandB sweep config created.
        """
        wandb.agent(
            self.sweep_id, self.step,
            count=self.num_trials
        )

        wandb._teardown()

        self.hpo_results.mark_finished()

        self.data_storer.save_hpo_result(self.hpo_results)

        return self.hpo_results
    
    