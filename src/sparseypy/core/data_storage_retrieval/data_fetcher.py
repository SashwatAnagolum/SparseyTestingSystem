# -*- coding: utf-8 -*-
"""
DataFetcher: Fetches data from weights and biases and the database (firestore)
"""
from datetime import datetime
from functools import lru_cache
import json
import os
import pickle
import torch

from firebase_admin import firestore
from google.api_core.datetime_helpers import DatetimeWithNanoseconds
import wandb

from sparseypy.core.data_storage_retrieval.data_storer import DataStorer
from sparseypy.core.db_adapters import DbAdapterFactory
from sparseypy.core.metrics import comparisons
from sparseypy.core.results.hpo_result import HPOResult
from sparseypy.core.results.hpo_step_result import HPOStepResult
from sparseypy.core.results.training_result import TrainingResult
from sparseypy.core.results.training_step_result import TrainingStepResult

class DataFetcher:
    """
    A class for fetching data from a Firestore database, including experiment data, 
    HPO run data, and model weights.

    This class provides methods to access and deserialize data related to Sparsey 
    experiments stored in Firestore. It supports caching for efficient data retrieval.
    """
    def __init__(self, config: dict):
        """
        Initializes the DataFetcher instance by setting up a connection to the Firestore database.
        (credentials need to have been set before using this)

        Args:
            config (dict): the system.yaml configuration.
        """
        if not DataStorer.is_initialized:
            raise ValueError("You must call DataStorer.configure() before intializing DataFetcher objects.")

        read_db_name = config["database"]["read_database"]

        read_config = next(
            (
                db for db in config["database"]["write_databases"]
                if db["name"] == read_db_name
            )
        )

        self.db_adapter = DbAdapterFactory.create_db_adapter(
            read_db_name,
            config=read_config,
            metric_config=[]
        )

        #self.tables = DataStorer.firestore_config["table_names"]

        #self.db = firestore.client()


    def get_model_source_path(self, model_name: str) -> str:
        """
        Retrieves the source experiment for a given model name and version.
        Args:
            model_name (str): A unique identifier for the model.
        Returns:
            str: the path to the run that trained this model version.
        """
        # construct the artifact name by adding ":latest" if the user has not
        # specified a version
        artifact_name = model_name + ('' if ':' in model_name else ':latest')
        # artifact path form: "<entity>/<project>/<artifact>"
        artifact_path = f"{wandb.api.default_entity}/model-registry/{artifact_name}"
        # fetch the artifact from W&B
        artifact = wandb.Api().artifact(artifact_path)
        return artifact.metadata["source_path"]


    def get_model_data(self, model_name: str) -> tuple[dict, dict]:
        """
        Fetches configuration and model weights for a given model.

        Args:
            model_name (str): A unique identifier for the model.

        Returns:
            dict: The contents of network.yaml for the model.
            dict: The state_dict containing the model weights.
        """
        # this uses WEIGHTS & BIASES to avoid problems with
        # model files larger than 1MB in Firestore

        # attempt to fetch the model directly from the registry
        # the model must be fetched as an artifact rather than using
        # use_model() because use_model() does not support fetching
        # from the model registry!

        # construct the artifact name by adding ":latest" if the user has not
        # specified a version
        artifact_name = model_name + ('' if ':' in model_name else ':latest')
        # artifact path form: "<entity>/<project>/<artifact>"
        artifact_path = f"{wandb.api.default_entity}/model-registry/{artifact_name}"
        # fetch the artifact from W&B
        m_ref = wandb.run.use_artifact(artifact_path, type="model")
        m_path = m_ref.download()
        # read the model config from the downloaded artifact
        with open(os.path.join(m_path, "network.yaml"), "r", encoding="utf-8") as f:
            model_config = json.load(f)
        # also load the state dict
        state_dict = torch.load(os.path.join(m_path, "model.pt"))

        return model_config, state_dict

    def get_training_step_result(
            self,
            experiment_id: str,
            step_index: int,
            result_type: str ="training"
        ) -> TrainingStepResult:
        """
        Retrieves the result of a specific training step within an experiment.

        Args:
            experiment_id (str): The unique identifier for the experiment.
            step_index (int): The index of the training step to retrieve.
            result_type (str): The type of result to retrieve. Defaults to "training".

        Returns:
            TrainingStepResult: An instance of TrainingStepResult containing the step's metrics.

        Raises:
            ValueError: If the step index is out of bounds for the given experiment.
        """
        return self.db_adapter.get_training_step_result(
            experiment_id=experiment_id,
            step_index=step_index,
            result_type=result_type
        )


    def get_evaluation_step_result(
            self,
            experiment_id: str,
            step_index: int,
        ) -> TrainingStepResult:
        """
        Retrieves the result of a specific evaluation step within an experiment.

        Args:
            experiment_id (str): The unique identifier for the experiment.
            step_index (int): The index of the training step to retrieve.

        Returns:
            TrainingStepResult: An instance of TrainingStepResult containing the 
                evaluation step's metrics.

        Raises:
            ValueError: If the step index is out of bounds for the given experiment.
        """
        return self.db_adapter.get_evaluation_step_result(
                experiment_id=experiment_id,
                step_index=step_index
            )


    def get_training_result(
            self,
            experiment_id: str,
            result_type: str = "training"
        ) -> TrainingResult:
        """
        Retrieves the training result for a given experiment.

        This method compiles the results of individual training steps within an experiment 
        into a single TrainingResult object. It includes overall metrics, step-by-step results, 
        and information about the start and end times of the experiment, as well as the 
        best performing steps.

        Args:
            experiment_id (str): The unique identifier for the experiment.

        Returns:
            TrainingResult: An instance of TrainingResult containing aggregated 
            metrics and outcomes from the experiment's training steps.
        """
        return self.db_adapter.get_training_result(experiment_id, result_type)
        # experiment_data = self._get_experiment_data(experiment_id)

        # metrics = []
        # tr = TrainingResult(id=experiment_id,
        #                     result_type=result_type,
        #                     resolution=experiment_data["saved_metrics"]["resolution"],
        #                     metrics=metrics,
        #                     configs={
        #                             conf_name:json.loads(conf_data)
        #                             for conf_name, conf_data in experiment_data["configs"]
        #                         }
        #                     )

        # for step_index in range(len(experiment_data.get("saved_metrics", {}).get(result_type, []))):
        #     step_result = self.get_training_step_result(experiment_id, step_index, result_type)
        #     tr.add_step(step_result)

        # tr.start_time = self.convert_firestore_timestamp(
        #         experiment_data.get("start_times", {}).get(result_type)
        #     )
        # tr.end_time = self.convert_firestore_timestamp(
        #         experiment_data.get("end_times", {}).get(result_type)
        #     )
        # best_steps = {}

        # phase_data = experiment_data.get("best_steps", {}).get(result_type, {})
        # best_steps = {}
        # for metric, metric_data in phase_data.items():
        #     best_function = metric_data.get("best_function")
        #     best_index = metric_data.get("best_index")
        #     best_value_bytes = metric_data.get("best_value")

        #     best_steps[metric] = {
        #         "best_function": getattr(comparisons, best_function),
        #         "best_index": best_index,
        #         "best_value": self._deserialize_metric(best_value_bytes)
        #     }
        # tr.best_steps = best_steps
        # return tr

    def get_evaluation_result(self, experiment_id: str) -> TrainingResult:
        """
        Get the evaluation result for a given experiment.

        Args:
            experiment_id (str): The ID of the experiment.

        Returns:
            EvaluationResult: the EvaluationResult for the experiment of this id in w&b
        """
        return self.db_adapter.get_evaluation_result(experiment_id=experiment_id)

    def get_hpo_step_result(self, hpo_run_id, experiment_id):
        """
        Retrieves the result of a specific experiment step within an HPO run.

        This method combines experiment data and HPO configuration to create a comprehensive
        step result for hpo.

        Args:
            hpo_run_id (str): The unique identifier for the HPO run.
            experiment_id (str): The unique identifier for the experiment within the HPO run.

        Returns:
            HPOStepResult: An instance of HPOStepResult representing the experiment step 
            within the HPO run.
        """
        return self.db_adapter.get_hpo_step_result(hpo_run_id, experiment_id)

    def get_hpo_result(self, hpo_run_id: str) -> HPOResult:
        """
        Retrieves the overall result of a specific hyperparameter optimization (HPO) run.

        This method aggregates the results of individual experiments within an HPO run, 
        and provides a comprehensive view of the HPO run, including start and end times, 
        configuration settings, and the best-performing experiment.

        Args:
            hpo_run_id (str): The unique identifier for the HPO run.

        Returns:
            HPOResult: An instance of HPOResult containing aggregated results 
            and configuration info from the HPO run.
        """
        return self.db_adapter.get_hpo_result(hpo_run_id)
        # hpo_run_data = self._get_hpo_run_data(hpo_run_id)

        # configs = {
        #     conf_name: json.loads(conf_json)
        #     for conf_name, conf_json in hpo_run_data["configs"].items()
        #     }
        # hpo_result = HPOResult(configs=configs, id=hpo_run_id, name=hpo_run_data["name"])

        # for experiment_id in hpo_run_data["runs"]:
        #     step_result = self.get_hpo_step_result(hpo_run_id, experiment_id)
        #     hpo_result.add_step(step_result)

        # hpo_result.best_run = self.get_hpo_step_result(hpo_run_id, hpo_run_data["best_run_id"])
        # hpo_result.start_time = self.convert_firestore_timestamp(hpo_run_data["start_time"])
        # hpo_result.end_time = self.convert_firestore_timestamp(hpo_run_data["end_time"])
        # return hpo_result

    # def convert_firestore_timestamp(self, firestore_timestamp: DatetimeWithNanoseconds) -> datetime:
    #     """
    #     Converts a Firestore DatetimeWithNanoseconds object to a standard Python datetime object.

    #     Args:
    #         firestore_timestamp (DatetimeWithNanoseconds): The Firestore timestamp to convert.

    #     Returns:
    #         datetime: A standard Python datetime object representing the same point in time.
    #     """
    #     converted_datetime = datetime(
    #         year=firestore_timestamp.year,
    #         month=firestore_timestamp.month,
    #         day=firestore_timestamp.day,
    #         hour=firestore_timestamp.hour,
    #         minute=firestore_timestamp.minute,
    #         second=firestore_timestamp.second,
    #         microsecond=firestore_timestamp.microsecond,
    #     )
    #     return converted_datetime
