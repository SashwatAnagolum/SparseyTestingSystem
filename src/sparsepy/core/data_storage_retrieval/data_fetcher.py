# -*- coding: utf-8 -*-
"""
DataFetcher: Fetches data from weights and biases
"""

import pickle
import json
from firebase_admin import firestore
from datetime import datetime
import wandb

from sparsepy.core.results.training_result import TrainingResult
from sparsepy.core.results.training_step_result import TrainingStepResult
from sparsepy.core.results.evaluation_result import EvaluationResult
from sparsepy.core.results.hpo_result import HPOResult
from sparsepy.core.results.hpo_step_result import HPOStepResult

class DataFetcher:
    """Fetches data related to sparsey models and results."""

    def __init__(self):
        
        # create API client
        self.api = wandb.Api()

        # connect to Firestore
        self.db = firestore.client()
    
    def get_model_weights(self, model_id: str) -> dict:
        """
        Fetches model weights for a given model ID.

        Args:
            model_id (str): A unique identifier for the model.

        Returns:
            dict: A dictionary of model weights.
        """
        pass
    
    def get_training_result(self, experiment_id: str) -> TrainingResult:
        """
        Gets the training result for a specific experiment.

        Args:
            experiment_id (str): The ID of the experiment.

        Returns:
            TrainingResult: This Experiment's TrainingResult from w&b
        """

        # fetch this experiment's data back from Firestore
        experiment_ref = self.db.collection("experiments").document(experiment_id)
        # this fetches ALL the data for the object and should be used sparingly
        experiment = experiment_ref.get().to_dict()

        # create a new TrainingResult object to hold the retrieved data
        tr = TrainingResult(experiment_id, experiment["saved_metrics"]["resolution"])

        # populate all the training steps
        for train_step in experiment["saved_metrics"]["training"]:
            tsr = TrainingStepResult(experiment["saved_metrics"]["resolution"])
            tsr.metrics = [
                {
                    met_name:pickle.loads(encoded_val) for met_name, encoded_val in train_step.items()
                }
            ]
            tr.add_step(tsr)

        # populate summary-level data: start, end, best steps
        tr.start_time = experiment["start_time"]
        tr.end_time = experiment["end_time"]
        # FIXME implement best step retrieval
        #tr.best_steps = experiment["best_steps"]

        return tr

    def get_eval_result(self, experiment_id: str) -> EvaluationResult:
        """
        Get the evaluation result for a given experiment.

        Args:
            experiment_id (str): The ID of the experiment.

        Returns:
            EvaluationResult: the EvaluationResult for the experiment of this id in w&b
        """
        pass
    
    def get_hpo_result(self, hpo_run_id: str) -> HPOResult:
        """
        Gets the HPO result for a specific HPO run.

        Args:
            hpo_run_id (str): The ID of the HPO run.

        Returns:
            HPOResult: This HPO run's HPOResult from the database
        """

        # Fetch this HPO run's data back from Firestore
        hpo_run_ref = self.db.collection("hpo_runs").document(hpo_run_id)
        hpo_run = hpo_run_ref.get().to_dict()

        # Parse configs from JSON to dict
        configs = {
            "dataset_config": json.loads(hpo_run["configs"]["dataset_config"]),
            "hpo_config": json.loads(hpo_run["configs"]["hpo_config"]),
            "preprocessing_config": json.loads(hpo_run["configs"]["preprocessing_config"]),
            "sweep_config": json.loads(hpo_run["configs"]["sweep_config"]),
            "training_recipe_config": json.loads(hpo_run["configs"]["training_recipe_config"])
        }

        # Reconstruct HPOResult object
        hpo_result = HPOResult(configs=configs, id=hpo_run_id, name=hpo_run["name"])

        # Set best_run_id from the data
        hpo_result.best_run = hpo_run["best_run_id"]
        # TODO ADD logic for reconstructing HPOStepResults
        # Fetch and add each HPOStepResult to the HPOResult object
        #for run_id in hpo_run["runs"]:
        #    step_result = self.get_hpo_step_result(run_id)
        #    hpo_result.add_step(step_result)

        # TODO convert to date time
        hpo_result.start_time = hpo_run["start_time"]
        hpo_result.end_time = hpo_run["end_time"]

        return hpo_result
