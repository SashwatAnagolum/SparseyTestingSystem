# -*- coding: utf-8 -*-

"""
Run HPO Task: script to run HPO.
"""


import os
from pprint import pprint
from tqdm import tqdm

from sparseypy.access_objects.hpo_runs.hpo_run  import HPORun
from sparseypy.core.data_storage_retrieval.data_storer import DataStorer


def run_hpo(hpo_config: dict,
            dataset_config: dict, preprocessing_config: dict,
            system_config: dict):
    """
    Runs hyperparameter optimization
    over the specified network hyperparameters
    to optimize for the specified objective.

    Args:
        hpo_config (dict): config info used to build the
            HPORun object.
        dataset_config (dict): config info used to build the
            dataset object.
        preprocessing_config (dict): config info used to build the
            preprocessing stack.
        system_config (dict): config info for the overall system
    """

    # silence WandB if requested by the user
    if system_config["wandb"]["silent"]:
        os.environ["WANDB_SILENT"] = "true"

    # initialize the DataStorer (logs into W&B and Firestore)
    DataStorer.configure(system_config)

    hpo_run = HPORun(
        hpo_config,
        dataset_config, preprocessing_config, system_config
    )

    # if we are in production mode (verbosity 0), suppress the W&B output
    if hpo_config["verbosity"] == 0:
        os.environ["WANDB_SILENT"] = "true"

    met_separator = "\n* "
    combination_item = "{mn:<25} (weight: {mw:.5f})"

    obj_vals = [
        combination_item.format(mn=x['metric']['name'], mw=x['weight'])
        for x in hpo_config['optimization_objective']['objective_terms']
        ]

    tqdm.write(f"""
HYPERPARAMETER OPTIMIZATION SUMMARY
          
W&B project name: {hpo_config['project_name']}
W&B sweep name: {hpo_config['hpo_run_name']}

Model family: {hpo_config['model_family']}
Optimization strategy: {hpo_config['hpo_strategy']}
Number of runs: {hpo_config['num_candidates']}

Selected metrics: 
* {met_separator.join([x["name"] for x in hpo_config["metrics"]])}

Objective calculation: {hpo_config['optimization_objective']['combination_method']} of
* {met_separator.join(obj_vals)}
""")

    hpo_results = hpo_run.run_sweep()

    time_delta = hpo_results.end_time - hpo_results.start_time

    print("\n---------------------------------------------------------")
    print("HYPERPARAMETER OPTIMIZATION COMPLETED")
    print(f"\nCompleted {hpo_config['num_candidates']} runs in {str(time_delta.seconds)} seconds.")
    # print the breakdown of the best-performing run
    print("\n---------------------------------------------------------")
    print(f"BEST RUN\n")
    hpo_run._print_breakdown(hpo_results.best_run, print_config=True)

    print("\n---------------------------------------------------------")
    print("Review full results in Weights & Biases:")
    print(f"Project: {hpo_config['project_name']}")
    print(f"HPO sweep name: {hpo_config['hpo_run_name']}")
    print(f"HPO sweep URL: {hpo_run.sweep_url}")
    print(f"Best run ID: {hpo_run.best.id}")
    print(f"Best run URL: {hpo_run.best_run_url}")
