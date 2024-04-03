# -*- coding: utf-8 -*-
"""
DataFetcher: Fetches data from weights and biases and the database (firestore)
"""

import json
import pickle
from datetime import datetime
from firebase_admin import firestore
import numpy as np
import wandb
from sparseypy.core.metrics import comparisons
from functools import lru_cache
from datetime import datetime
from google.api_core.datetime_helpers import DatetimeWithNanoseconds

from sparseypy.core.results.training_result import TrainingResult
from sparseypy.core.results.training_step_result import TrainingStepResult
from sparseypy.core.results.hpo_result import HPOResult
from sparseypy.core.results.hpo_step_result import HPOStepResult

class DataFetcher:
    """
    A class for fetching data from a Firestore database, including experiment data, HPO run data, and model weights.

    This class provides methods to access and deserialize data related to sparsey experiments stored in Firestore.
    It supports caching for efficient data retrieval.
    """
    def __init__(self):
        """
        Initializes the DataFetcher instance by setting up a connection to the Firestore database.
        (credentials need to have been set before using this)
        """
        self.db = firestore.client()
    
    def _deserialize_metric(self, serialized_metric):
        """
        Deserializes a metric value stored as a pickled string.

        Args:
            serialized_metric (bytes): The pickled representation of a metric.

        Returns:
            object: The deserialized metric value.
        """
        return pickle.loads(serialized_metric)

    @lru_cache(maxsize=None)
    def _get_experiment_data(self, experiment_id):
        """
        Retrieves and caches the data for a specific experiment from Firestore.

        Args:
            experiment_id (str): The unique identifier for the experiment.

        Returns:
            dict: A dictionary containing the experiment data.
        """
        experiment_ref = self.db.collection("experiments").document(experiment_id)
        return experiment_ref.get().to_dict()

    @lru_cache(maxsize=None)
    def _get_hpo_run_data(self, hpo_run_id):
        """
        Retrieves and caches the data for a specific HPO run from Firestore.

        Args:
            hpo_run_id (str): The unique identifier for the HPO run.

        Returns:
            dict: A dictionary containing the HPO run data.
        """
        hpo_run_ref = self.db.collection("hpo_runs").document(hpo_run_id)
        return hpo_run_ref.get().to_dict()

    def get_model_weights(self, model_id: str) -> dict:
        """
        Fetches model weights for a given model ID.

        Args:
            model_id (str): A unique identifier for the model.

        Returns:
            dict: A dictionary of model weights.
        """
        pass

    def get_training_step_result(self, experiment_id, step_index):
        """
        Retrieves the result of a specific training step within an experiment.

        Args:
            experiment_id (str): The unique identifier for the experiment.
            step_index (int): The index of the training step to retrieve.

        Returns:
            TrainingStepResult: An instance of TrainingStepResult containing the step's metrics.

        Raises:
            ValueError: If the step index is out of bounds for the given experiment.
        """
        experiment_data = self._get_experiment_data(experiment_id)
        training_steps = experiment_data.get("saved_metrics", {}).get("training", [])
        if step_index < 0 or step_index >= len(training_steps):
            raise ValueError("Step index is out of bounds for the given experiment.")

        step_data = training_steps[step_index]
        step_result = TrainingStepResult(resolution=experiment_data["saved_metrics"]["resolution"])
        
        for metric_name, metric_data in step_data.items():
            step_result.add_metric(name=metric_name, values=self._deserialize_metric(metric_data))

        return step_result

    def get_training_result(self, experiment_id: str, result_type: str = "training") -> TrainingResult:
        """
        Retrieves the training result for a given experiment.

        This method compiles the results of individual training steps within an experiment into a single TrainingResult object.
        It includes overall metrics, step-by-step results, and information about the start and end times of the experiment,
        as well as the best performing steps.

        Args:
            experiment_id (str): The unique identifier for the experiment.

        Returns:
            TrainingResult: An instance of TrainingResult containing aggregated metrics and outcomes from the experiment's training steps.
        """
        experiment_data = self._get_experiment_data(experiment_id)

        metrics = []
        tr = TrainingResult(id=experiment_id,
                            result_type=result_type,
                            resolution=experiment_data["saved_metrics"]["resolution"],
                            metrics=metrics,
                            configs={
                                    conf_name:json.loads(conf_data)
                                    for conf_name, conf_data in experiment_data["configs"]
                                }
                            )

        for step_index in range(len(experiment_data.get("saved_metrics", {}).get(result_type, []))):
            step_result = self.get_training_step_result(experiment_id, step_index)
            tr.add_step(step_result)

        tr.start_time = self.convert_firestore_timestamp(experiment_data.get("start_times", {}).get(result_type))
        tr.end_time = self.convert_firestore_timestamp(experiment_data.get("end_times", {}).get(result_type))
        best_steps = {}
        
        phase_data = experiment_data.get("best_steps", {}).get(result_type, {})
        best_steps = {}
        for metric, metric_data in phase_data.items():
            best_function = metric_data.get("best_function")
            best_index = metric_data.get("best_index")
            best_value_bytes = metric_data.get("best_value")

            best_steps[metric] = {
                "best_function": getattr(comparisons, best_function),
                "best_index": best_index,
                "best_value": self._deserialize_metric(best_value_bytes)
            }
        tr.best_steps = best_steps
        return tr
    
    def get_evaluation_result(self, experiment_id: str) -> TrainingResult:
        """
        Get the evaluation result for a given experiment.

        Args:
            experiment_id (str): The ID of the experiment.

        Returns:
            EvaluationResult: the EvaluationResult for the experiment of this id in w&b
        """
        return self.get_training_result(experiment_id=experiment_id, result_type="evaluation")

    def get_hpo_step_result(self, hpo_run_id, experiment_id):
        """
        Retrieves the result of a specific experiment step within an HPO run.

        This method combines experiment data and HPO configuration to create a comprehensive
        step result for hpo.

        Args:
            hpo_run_id (str): The unique identifier for the HPO run.
            experiment_id (str): The unique identifier for the experiment within the HPO run.

        Returns:
            HPOStepResult: An instance of HPOStepResult representing the experiment step within the HPO run.
        """
        # Assuming this method will utilize get_training_result to fetch associated training and evaluation results
        experiment_data = self._get_experiment_data(experiment_id)
        training_result = self.get_training_result(experiment_id)
        evaluation_result = self.get_evaluation_result(experiment_id)
        hpo_run_data = self._get_hpo_run_data(hpo_run_id)
        # TODO Populate model configs
        # TODO Check if the configs will be stored in experiments
        hpo_step_result = HPOStepResult(
            parent_run=hpo_run_id,
            id=experiment_id,
            configs={
                conf_name:json.loads(conf_json)
                for conf_name, conf_json in hpo_run_data["configs"].items()
                }
            )
        hpo_step_result.populate(
                objective=experiment_data["hpo_objective"],
                training_results=training_result,
                eval_results=evaluation_result
            )
        return hpo_step_result

    def get_hpo_result(self, hpo_run_id: str) -> HPOResult:
        """
        Retrieves the overall result of a specific hyperparameter optimization (HPO) run.

        This method aggregates the results of individual experiments within an HPO run, and provides a comprehensive 
        view of the HPO run, including start and end times, configuration settings, and the best-performing experiment.

        Args:
            hpo_run_id (str): The unique identifier for the HPO run.

        Returns:
            HPOResult: An instance of HPOResult containing aggregated results and configuration info from the HPO run.
        """
        hpo_run_data = self._get_hpo_run_data(hpo_run_id)

        configs = {conf_name: json.loads(conf_json) for conf_name, conf_json in hpo_run_data["configs"].items()}
        hpo_result = HPOResult(configs=configs, id=hpo_run_id, name=hpo_run_data["name"])

        for experiment_id in hpo_run_data["runs"]:
            step_result = self.get_hpo_step_result(hpo_run_id, experiment_id)
            hpo_result.add_step(step_result)
            
        hpo_result.best_run = self.get_hpo_step_result(hpo_run_id, hpo_run_data["best_run_id"])
        hpo_result.start_time = self.convert_firestore_timestamp(hpo_run_data["start_time"])
        hpo_result.end_time = self.convert_firestore_timestamp(hpo_run_data["end_time"])
        return hpo_result
    
    def convert_firestore_timestamp(self, firestore_timestamp: DatetimeWithNanoseconds) -> datetime:
        """
        Converts a Firestore DatetimeWithNanoseconds object to a standard Python datetime object.

        Args:
            firestore_timestamp (DatetimeWithNanoseconds): The Firestore timestamp to convert.

        Returns:
            datetime: A standard Python datetime object representing the same point in time.
        """
        converted_datetime = datetime(
            year=firestore_timestamp.year,
            month=firestore_timestamp.month,
            day=firestore_timestamp.day,
            hour=firestore_timestamp.hour,
            minute=firestore_timestamp.minute,
            second=firestore_timestamp.second,
            microsecond=firestore_timestamp.microsecond,
        )
        return converted_datetime
