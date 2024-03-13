# -*- coding: utf-8 -*-
"""
DataFetcher: Fetches data from weights and biases
"""
from sparsepy.core.results.training_result import TrainingResult
from sparsepy.core.results.evaluation_result import EvaluationResult
from sparsepy.core.results.hpo_result import HPOResult

class DataFetcher:
    """Fetches data related to sparsey models and results."""
    
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
        pass

    def get_eval_result(self, experiment_id: str) -> EvaluationResult:
        """
        Get the evaluation result for a given experiment.

        Args:
            experiment_id (str): The ID of the experiment.

        Returns:
            EvaluationResult: the EvaluationResult for the experiment of this id in w&b
        """
        pass
    
    def get_hpo_result(self, experiment_id: str) -> HPOResult:
        """
        Get the hyperparameter optimization result for an experiment.

        Args:
            experiment_id (str): The ID of the experiment.

        Returns:
            HPOResult: the HPOResult for the experiment of this id in w&b
        """
        pass
