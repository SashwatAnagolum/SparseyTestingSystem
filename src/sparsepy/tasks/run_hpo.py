# -*- coding: utf-8 -*-

"""
Run HPO Task: script to run HPO.
"""


from typing import Optional

import wandb

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

    hpo_run.run_sweep()
