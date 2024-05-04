# -*- coding: utf-8 -*-

"""
Run HPO Task: script to run HPO.
"""


import os
import shutil
from tqdm import tqdm
import warnings

from sparseypy.access_objects.hpo_runs.hpo_run  import HPORun
from sparseypy.core.data_storage_retrieval.data_storer import DataStorer
from sparseypy.core.printing import Printer

# filter out PyTorch warnings for using NestedTensors
warnings.filterwarnings(
    "ignore",
    message=r"The PyTorch API of nested tensors is in prototype stage and will change in the.+"
)

def run_hpo(hpo_config: dict,
            training_dataset_config: dict,
            evaluation_dataset_config: dict,
            preprocessing_config: dict,
            system_config: dict) -> None:
    """
    Runs hyperparameter optimization
    over the specified network hyperparameters
    to optimize for the specified objective.

    Args:
        hpo_config (dict): config info used to build the
            HPORun object.
        training_dataset_config (dict): config info used to
            build the training dataset object.
        evaluation_dataset_config (dict): config info used to
            build the evaluation dataset object.
        preprocessing_config (dict): config info used to build the
            preprocessing stack.
        system_config (dict): config info for the overall system
    """

    # silence WandB if requested by the user
    if system_config["wandb"]["silent"]:
        os.environ["WANDB_SILENT"] = "true"

    # initialize the DataStorer (logs into W&B and Firestore)
    tqdm.write("Connecting to Weights & Biases...")
    DataStorer.configure(system_config)

    hpo_run = HPORun(
        hpo_config, training_dataset_config,
        evaluation_dataset_config, preprocessing_config, 
        system_config
    )

    # if we are in production mode (verbosity 0), suppress the W&B output
    if hpo_config["verbosity"] == 0:
        os.environ["WANDB_SILENT"] = "true"

    # print the HPO summary
    Printer.print_pre_hpo_summary(hpo_config)

    hpo_results = hpo_run.run_sweep()

    Printer.print_post_hpo_summary(hpo_results, hpo_config,
                                   hpo_run.sweep_url, hpo_run.best_run_url)

    # remove results at completion if requested
    if system_config['wandb'].get('remove_local_files', False):
        for wandb_dir in hpo_run.wandb_dirs:
            shutil.rmtree(wandb_dir)
        tqdm.write("\nRemoved local temporary files.")
