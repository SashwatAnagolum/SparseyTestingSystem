# -*- coding: utf-8 -*-

"""
Run HPO Task: script to run HPO.
"""


from typing import Optional

import os
import wandb
from pprint import pprint
from sparsepy.access_objects.hpo_runs.hpo_run  import HPORun


def run_hpo(hpo_config: dict, trainer_config: dict,
            dataset_config: dict, preprocessing_config: dict,
            wandb_api_key: Optional[str] = ''):
    """
    Runs hyperparameter optimization
    over the specified network hyperparameters
    to optimize for the specified objective.

    Args:
        hpo_config (dict): config info used to build the
            HPORun object.
        trainer_config (dict): config info used to build the 
            trainer.
        dataset_config (dict): config info used to build the
            dataset object.
        preprocessing_config (dict): config info used to build the
            preprocessing stack.
        wandb_api_key (str): the Weights and Biases API key to use
            to log information to Weights and Biases.
    """
    hpo_run = HPORun(
        hpo_config, trainer_config,
        dataset_config, preprocessing_config,
        wandb_api_key
    )

    # if we are in production mode (verbosity 0), suppress the W&B output
    if hpo_config["verbosity"] == 0:
        os.environ["WANDB_SILENT"] = "true"

    hpo_results = hpo_run.run_sweep()
    print(f"OPTIMIZATION RUN COMPLETED")
    print(f"Best run: {hpo_results.best_run.id}")
    hpo_run._print_breakdown(hpo_results.best_run)
    print(f"Best run configuration:\n---------------------------------------------------------")
    layer_number = 1
    print('INPUT DIMENSIONS ')
    pprint(hpo_results.best_run.configs["model_config"]["input_dimensions"])
    print("\n---------------------------------------------------------")
    for layer in hpo_results.best_run.configs["model_config"]["layers"]:
        print("LAYER ", layer_number, "\n---------------------------------------------------------")
        pprint(layer)
        layer_number+=1
