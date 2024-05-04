# -*- coding: utf-8 -*-

"""
HPO Run: file holding the HPORun class.
"""

from copy import deepcopy
import os
from pprint import pformat
import traceback
import warnings

from tqdm import tqdm
import wandb

from sparseypy.access_objects.training_recipes.training_recipe_builder import TrainingRecipeBuilder
from sparseypy.cli.config_validation.validate_config import validate_config
from sparseypy.core.data_storage_retrieval import DataStorer
from sparseypy.access_objects.datasets.dataset_factory import DatasetFactory
from sparseypy.core.hpo_objectives.hpo_objective import HPOObjective
from sparseypy.core.results import HPOResult, HPOStepResult
from sparseypy.core.printing import Printer

# Weights & Biases attempts to read tqdm updates from the console even after the last run
# in an HPO sweep finishes, causing an unnecessary UserWarning when it attempts to log data
# to a nonexistent run; this is a Weights & Biases issue that does not affect system
# functionality so we ignore this warning
warnings.filterwarnings(
    "ignore",
    message="Run (.*) is finished. The call to `_console_raw_callback` will be ignored."
    )
# PyTorch nested tensors are in beta and print a warning about API changes
warnings.filterwarnings(
    "ignore",
    message="The PyTorch API of nested tensors is in prototype stage and will change in the near future."
)

class HPORun():
    """
    HPORun: class for performing HPO Runs.

    Attributes:
        num_steps_to_perform (int): the total number of 
            candidates to try out during the HPO process
    """
    tqdm_bar = None

    def __init__(self, hpo_config: dict, training_dataset_config: dict,
        evaluation_dataset_config: dict,
        preprocessing_config: dict, system_config: dict) -> None:
        """
        Initializes the HPORun object.

        Args:
            hpo_config (dict): configurations for the HPO Run.
            trainer_config (dict): configruations for the
                training recipe.
            training_dataset_config (dict): configurations for
                the training dataset.
            evaluation_dataset_config (dict): configurations for
                the evaluation dataset.
            preprocessing_config (dict): configurations for the
                preprocessing stack.
            wandb_api_key (str): the Weights and Biases API key to 
                use to login to WandB and log data.
        """
        self.sweep_config = self.construct_sweep_config(hpo_config, system_config)
        self.sweep_id = wandb.sweep(sweep=self.sweep_config)
        self.num_trials = hpo_config['num_candidates']
        self.config_info = hpo_config

        #trainer_config = hpo_config['trainer']

        self.system_config = system_config  
        self.preprocessing_config = preprocessing_config
        self.training_dataset_config = training_dataset_config
        self.evaluation_dataset_config = evaluation_dataset_config

        self.training_dataset = DatasetFactory.build_and_wrap_dataset(
            training_dataset_config
        )

        self.evaluation_dataset = DatasetFactory.build_and_wrap_dataset(
            evaluation_dataset_config
        )

        logged_configs = {
            'hpo_config': hpo_config,
            'sweep_config': self.sweep_config,
            'training_dataset_config': training_dataset_config,
            'evaluation_dataset_config': evaluation_dataset_config,
            'preprocessing_config': preprocessing_config
        }

        # create the DataStorer
        self.data_storer = DataStorer(hpo_config['metrics'])

        # create the HPOResult (also sets start time)
        self.hpo_results = HPOResult(
            logged_configs, self.sweep_id, hpo_config['hpo_run_name']
        )

        # create the sweep
        self.data_storer.create_hpo_sweep(self.hpo_results)

        # save the sweep URL
        entity = system_config['wandb'].get('entity', None)
        entity_locator = f"{entity}/" if entity else ""
        locator = f"{entity_locator}{hpo_config['project_name']}/{self.sweep_id}"
        self.sweep_url = wandb.Api().sweep(locator).url
        self.best_run_url = None

        # start the list of temporary directories with this sweep's temp dir
        local_dir = system_config['wandb']['local_log_directory']
        if local_dir is None:
            local_dir = "."

        self.wandb_dirs = [
            os.path.join(local_dir, 'wandb', 'sweep-' + self.sweep_id)
        ]

        # only initialize the objective once, in the constructor
        self.objective = HPOObjective(hpo_config)
        self.best = None
        self.num_steps = 0
        self.progress_bars = system_config['console']['hpo_progress_bars']
        if HPORun.tqdm_bar is None:
            HPORun.tqdm_bar = tqdm(
                total=self.num_trials,
                desc="HPO Trials",
                position=1,
                unit="trial"
            )


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


    def construct_sweep_config(self, hpo_config: dict,
        system_config: dict) -> dict:
        """
        Construct the sweep configuration for the Weights and Biases
        sweep to be performed as part of the HPO run.

        Args:
            hpo_config (dict): configuration info for the HPO run.
            system_config (dict): system configuration information.

        Returns:
            (dict): the WandB sweep configuration.
        """
        sweep_hyperparams = self.extract_hyperparams(
            hpo_config['hyperparameters']
        )

        sweep_config = {
            'description': hpo_config['description'],
            'entity': system_config['wandb']['entity'],
            'method': hpo_config['hpo_strategy'],
            'metric': {'goal': 'minimize', 'name': 'hpo_objective'},
            'name': hpo_config['hpo_run_name'],
            'project': hpo_config['project_name'],
            'run_cap': hpo_config['num_candidates'],
            'parameters': sweep_hyperparams
        }

        return sweep_config


    def generate_model_config(self, wandb_config: dict) -> dict:
        """
        Generate the model configuration for the next run
        to be performed as part of the sweep.

        Args:
            wandb_config (dict): the Weights & Biases configuration
                for the current run in the sweep

        Returns:
            dict: the model configuration in the system format
        """
        model_config = dict()
        layer_keys = dict()
        layers = []

        for key, value in wandb_config.items():
            if 'layers_' in key:
                layer_keys[key] = value
            elif key != 'num_layers' and key != 'trainer':
                model_config[key] = value

        for i in range(wandb_config['num_layers']):
            layers.append(layer_keys[f'layers_{i}'])

        model_config['layers'] = layers

        return model_config


    def generate_trainer_config(self, wandb_config: dict) -> dict:
        """
        Generate the trainer configuration for the next run
        to be performed as part of the sweep.

        Args:
            wandb_config (dict): the Weights & Biases configuration
                for the current run in the sweep.

        Returns:
            dict: the trainer configuration in the system format
        """
        # get the trainer hyperparameters from the W&B config
        train_config = wandb_config["trainer"]
        # inject the metric list from the HPO config to complete the trainer config
        train_config["metrics"] = self.config_info["metrics"]
        # also inject the GPU configuration
        train_config["use_gpu"] = self.config_info["use_gpu"]

        return train_config


    def step(self) -> None:
        """
        Perform one HPO step, which includes sampling 
        a set of model hyperparameters, training the created
        model, and computing the user-specified objective function
        using the trained model.
        """
        wandb.init(
            allow_val_change=True,
            dir=self.system_config['wandb']['local_log_directory'],
            job_type="train"
        )

        model_config = self.generate_model_config(
            dict(wandb.config)
        )

        trainer_config = self.generate_trainer_config(
            dict(wandb.config)
        )

        validated_model_config = validate_config(
            model_config, 'model', self.config_info['model_family'],
            survive_with_exception=True,
            print_error_stacktrace=self.system_config['console']['print_error_stacktrace']
        )

        validated_trainer_config = validate_config(
            trainer_config, 'training_recipe', 'sparsey',
            survive_with_exception=True,
            print_error_stacktrace=self.system_config['console']['print_error_stacktrace']
        )

        try:
            training_recipe = TrainingRecipeBuilder.build_training_recipe(
                model_config=validated_model_config,
                training_dataset_config=self.training_dataset_config,
                evaluation_dataset_config=self.evaluation_dataset_config,
                preprocessing_config=self.preprocessing_config,
                train_config=validated_trainer_config,
                training_dataset=self.training_dataset,
                evaluation_dataset=self.evaluation_dataset
            )

            done = False
            results = None

            # do we need to move this earlier?
            hpo_step_results = HPOStepResult(
                parent_run=self.sweep_id, id=wandb.run.id,
                configs={
                    'training_dataset_config': self.training_dataset_config,
                    'evaluation_dataset_config': self.evaluation_dataset_config,
                    'preprocessing_config': self.preprocessing_config,
                    'training_recipe_config': validated_trainer_config,
                    'model_config': validated_model_config
                }
            )

            # increment step counter
            self.num_steps += 1

            # perform training
            with tqdm(
                total=training_recipe.training_num_batches,
                desc=f"Training (Trial {self.num_steps})",
                leave=False, position=0,
                disable=(not self.progress_bars),
                unit="input",
                miniters=int(training_recipe.training_num_batches/100)
            ) as pbar:
                while not done:
                    results, done = training_recipe.step()
                    pbar.update(1)
            # fetch training results
            training_results = training_recipe.get_summary("training")
            # perform evaluation
            done = False
            with tqdm(
                total=training_recipe.eval_num_batches,
                desc=f"Evaluation (Trial {self.num_steps})",
                leave=False, position=0,
                disable=(not self.progress_bars),
                unit="input",
                miniters=int(training_recipe.eval_num_batches/100)
            ) as pbar:
                while not done:
                    results, done = training_recipe.step(training=False)
                    pbar.update(1)
            # fetch evaluation results
            eval_results = training_recipe.get_summary("evaluation")

            # calculate the objective from the evaluation results
            objective_results = self.objective.combine_metrics(eval_results)

            if results is not None:
                # final result ready

                # populate the HPOStepResult
                hpo_step_results.populate(
                    objective=objective_results,
                    training_results=training_results,
                    eval_results=eval_results
                )

                # add the HPOStepResults to the HPOResult
                self.hpo_results.add_step(hpo_step_results)

                # this step is the best step if 1) there is no previous result or
                # 2) its objective value is higher than the previous best result
                new_best = (
                    not self.best or
                    (objective_results["total"] > self.best.get_objective()["total"])
                   )

                if new_best:
                    self.best = hpo_step_results
                    self.best_run_url = wandb.run.url

                # print the step summary
                Printer.summarize_hpo_trial(
                    step_results=hpo_step_results,
                    step_num=self.num_steps,
                    num_trials=self.num_trials,
                    print_config=False,
                    new_best=new_best
                )

                self.data_storer.save_hpo_step(wandb.run.sweep_id, hpo_step_results)

                # cache run path for updating config
                run_path = wandb.run.path
                # add temporary directory to removal list
                self.wandb_dirs.append(wandb.run.dir.removesuffix("files"))

                # finish the run - wandb.run may no longer be correct below this point
                wandb.finish()

                # strip unused layers from W&B side config
                # this must occur after .finish() due to a bug in W&B preventing
                # config file changes during a run even with allow_val_changes
                max_layers = len(model_config['layers'])
                run = wandb.Api().run(run_path)

                for k in run.config.copy():
                    if ("layers_" in k) and (int(k[7:]) >= max_layers):
                        del run.config[k]

                run.update()
        except Exception as e:
            Printer.print_hpo_exception(
                current_step=self.num_steps,
                message=traceback.format_exc()
            )

        if HPORun.tqdm_bar is not None:
            HPORun.tqdm_bar.update(1)


    @classmethod
    def close_tqdm(cls):
        """
        Closes the tqdm progress bar, if it exists.
        """
        if cls.tqdm_bar is not None:
            cls.tqdm_bar.close()
            cls.tqdm_bar = None


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
        HPORun.close_tqdm()

        return self.hpo_results
