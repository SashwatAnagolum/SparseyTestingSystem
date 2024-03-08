# -*- coding: utf-8 -*-
"""
DataFetcher: Fetches data from weights and biases
"""

import pickle

from firebase_admin import firestore

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
    
    def get_hpo_result(self, experiment_id: str) -> HPOResult:
        """
        Get the hyperparameter optimization result for an experiment.

        Args:
            experiment_id (str): The ID of the experiment.

        Returns:
            HPOResult: the HPOResult for the experiment of this id in w&b
        """
        pass
