# -*- coding: utf-8 -*-
"""
DataStorer: Saves data to Weights & Biases and the system database (Firestore)
"""
import json
import os
import pickle
import tempfile

import firebase_admin
from firebase_admin import firestore # firestore_async for async client
import numpy as np
import torch
import wandb

from sparseypy.core.results.training_result import TrainingResult
from sparseypy.core.results.training_result import TrainingStepResult
from sparseypy.core.results.hpo_result import HPOResult
from sparseypy.core.results.hpo_step_result import HPOStepResult
from sparseypy.access_objects.models.model import Model


class DataStorer:
    """
    DataStorer: Stores data to Weights & Biases
    """
    # static variables
    is_initialized = False
    wandb_config = {}
    firestore_config = {}


    def __init__(self, metric_config: dict):
        # configure saved metrics?
        self.saved_metrics = [metric["name"] for metric in metric_config if metric["save"] is True]
        # create API client
        self.api = wandb.Api()

        # connect to Firestore
        self.db = firestore.client()

        self.wandb_resolution = DataStorer.wandb_config["data_resolution"]
        self.firestore_resolution = DataStorer.firestore_config["data_resolution"]

    @staticmethod
    def configure(ds_config: dict):
        """
        Configures the DataStorer by logging into Weights & Biases and
        initializing its database connection.

        Because all configuration is tracked inside firebase_admin and 
        wandb, calling this method also configures the DataFetcher.

        Args:
            ds_config (dict): the validated system.yaml configuration
        """
        if not DataStorer.is_initialized:

            # initialize W&B
            wandb.login(key=ds_config['wandb']['api_key'], verify=True)

            # initialize Firestore
            cred_obj = firebase_admin.credentials.Certificate(
                ds_config['database']['write_databases'][0]['firebase_service_key_path']
                )
            firebase_admin.initialize_app(cred_obj)

            DataStorer.wandb_config = ds_config['wandb']
            DataStorer.firestore_config = ds_config["database"]["write_databases"][0]

            DataStorer.is_initialized = True

    def save_model(self, experiment: str, m: Model):
        """
        Saves a model to Weights & Biases.

        Args:
            experiment (str): the experiment ID to which the model should be saved
            m (Model): the model object to be saved
        """
        # Implementation to save the model
        pass

    def save_training_step(self, parent: str, result: TrainingStepResult):
        """
        Saves a single training step to Weights & Biases and Firestore.

        Args:
            parent (str): the experiment ID to which to log this step
            result (TrainingStepResult): the step results to save
        """
        summary_dict = {}
        full_dict = {}

        # gather the saved metrics for summary in W&B and full storage in the DB
        for metric_name, metric_val in result.get_metrics().items():
            if metric_name in self.saved_metrics:
                summary_dict[metric_name] = self.average_nested_data(metric_val)
                full_dict[metric_name] = pickle.dumps(metric_val) # pickling

        # log layerwise data to W&B, if requested
        if self.wandb_resolution > 0:
            layerwise_dict = {}
            # get the data for each metric
            for metric_name, metric_val in result.get_metrics().items():
                # if that metric is requested for saving and is at least 1D in layers (is a list)
                if metric_name in self.saved_metrics and isinstance(metric_val, list):
                    # then break out each layer as a separate metric for W&B using prefix grouping
                    for idx, layer_data in enumerate(metric_val):
                        layer_name = f"{metric_name}/layer_{idx}"
                        layerwise_dict[layer_name] = self.average_nested_data(layer_data)
                        # if the resolution is 2 (MAC-level)
                        # also log the MAC-level data with prefix grouping
                        if self.wandb_resolution == 2 and isinstance(layer_data, list):
                            for idy, mac_data in enumerate(layer_data):
                                mac_name = f"{metric_name}/layer_{idx}/mac_{idy}"
                                layerwise_dict[mac_name] = self.average_nested_data(mac_data)
            # then log without updating the step (done when the summary is logged below)
            wandb.log(layerwise_dict, commit=False)

        # save the summary to W&B
        # add this if we add step resolution back now that we don't need W&B concordance
        #summary_dict["resolution"] = result.resolution
        wandb.log(summary_dict)

        # save to Firestore on "step" resolution only
        if self.firestore_resolution == 2:

            # save the full results to Firestore
            experiment_ref = self.db.collection("experiments").document(parent)

            if experiment_ref.get().exists:
                # add step to existing experiment
                experiment_ref.update(
                    {
                        "saved_metrics.training": firestore.ArrayUnion([full_dict])
                    }
                )
            #else:
            # raise exception

    def save_evaluation_step(self, parent: str, result: TrainingStepResult):
        """
        Saves a single evaluation step to Weights & Biases and Firestore.

        Args:
            parent (str): the experiment ID to which to log this step
            result (TrainingStepResult): the step results to save
        """
        # gather the saved metrics for storage in the DB
        full_dict = {}
        for metric_name, metric_val in result.get_metrics().items():
            if metric_name in self.saved_metrics:
                full_dict[metric_name] = pickle.dumps(metric_val) # pickling

        # save to Firestore on "step" resolution only
        if self.firestore_resolution == 2:

            # save the full results to Firestore
            experiment_ref = self.db.collection("experiments").document(parent)

            if experiment_ref.get().exists:
                # add step to existing experiment
                experiment_ref.update(
                    {
                        "saved_metrics.evaluation": firestore.ArrayUnion([full_dict])
                    }
                )
            #else:
            # raise exception

    def create_experiment(self, experiment: TrainingResult):
        """
        Creates a new entry for the current experiment in Firestore.
        
        Args:
            experiment (TrainingResult): the TrainingResult for the new experiment
            for which to create a database entry
        """
        # save on "summary" or better
        if self.firestore_resolution > 0:
            # create the DB entry for this experiment in Firestore
            experiment_ref = self.db.collection("experiments").document(experiment.id)

            experiment_ref.set(
                {
                    "start_times": {
                        "experiment": experiment.start_time,
                        "training": experiment.start_time
                    },
                    "saved_metrics": {
                        "resolution": experiment.resolution
                    },
                    "end_times": {},
                    "completed": False
                }
            )

    def save_training_result(self, result: TrainingResult):
        """
        Saves the summary-level training results for the current run
        to Firestore. 

        Only saves the training summary--you still need to save the individual 
        training steps by calling save_training_step().

        Args:
            result (TrainingResult): the completed training results
            to save
        """
        # Implementation to save the training result

        # do we even need to set anything in W&B here? time finished? but W&B should track that
        # is there something we need to do here to mark end of run?

        # save on "summary" or better
        if self.firestore_resolution > 0:

            experiment_ref = self.db.collection("experiments").document(result.id)

            # every invocation of this costs 1 read call; consider removing
            if experiment_ref.get().exists:
                experiment_ref.update(
                    {
                        "start_times.training": result.start_time,
                        "end_times.training": result.end_time,
                        "completed": True,
                        "best_steps.training": {
                                metric_name:{
                                    'best_index': metric_vals["best_index"],
                                    'best_value': pickle.dumps(metric_vals["best_value"]),
                                    'best_function': metric_vals["best_function"].__name__} 
                                for metric_name, metric_vals in result.best_steps.items()
                            },
                        "configs": {
                            conf_name: json.dumps(conf_data)
                            for conf_name, conf_data in result.configs.items()
                        }
                    }
                )

        # COMMENT THIS IN to test updated config saving
        #run.config = result.

    def save_evaluation_result(self, result: TrainingResult):
        """
        Saves the summary-level evaluation results for the current run
        to Firestore. 

        Only saves the evaluation summary--you still need to save the individual 
        evaluation steps by calling save_evaluation_step().

        Args:
            result (TrainingResult): the completed evaluation results
            to save
        """
        # WEIGHTS & BIASES

        # access the current run's summary-level data with the API
        run = self.api.run(wandb.run.path)

        if self.firestore_resolution > 0:

            # gather the saved metrics for summary in W&B
            eval_dict = {
                # create a key for each saved metric containing the nested average
                # of the results of the metric for each step in the evaluation
                saved_metric:self.average_nested_data(
                    [step.get_metric(saved_metric) for step in result.get_steps()]
                                                      ) for saved_metric in self.saved_metrics
            }
            # then add those as "evaluation_" results to the W&B summary level
            for metric_name, metric_val in eval_dict.items():
                run.summary["evaluation_" + metric_name] = metric_val
            # save summary to W&B
            run.summary.update()
            # save the full results to Firestore
            experiment_ref = self.db.collection("experiments").document(result.id)
            if experiment_ref.get().exists:
                # add step to existing experiment
                experiment_ref.update(
                    {
                        "start_times.evaluation": result.start_time,
                        "end_times.evaluation": result.end_time,
                        "end_times.experiment": result.end_time,
                        "completed": True,
                        "best_steps.evaluation": {
                                metric_name:{
                                    'best_index': metric_vals["best_index"],
                                    'best_value': pickle.dumps(metric_vals["best_value"]),
                                    'best_function': metric_vals["best_function"].__name__} 
                                for metric_name, metric_vals in result.best_steps.items()
                            }
                    }
                )

    def save_hpo_step(self, parent: str, result: HPOStepResult):
        """
        Saves a single HPO step to Weights & Biases and Firestore.

        Saves objective data and HPO configuration to the run in
        both Weights & Biases and Firestore.

        Also marks this experiment in Firestore as belonging to the
        parent sweep and updates its best runs.
        
        Args:
            parent (str): the ID of the parent sweep in the HPO table
            that should be updated with this run's results
            result (HPOStepResult): the results of the HPO step to save
        """
        # WEIGHTS & BIASES
        # save the objective and HPO config to this experiment

        objective = result.get_objective()
        # FIXME correct duplicate logging, only log hpo_objective
        wandb.log(
                    {
                    'hpo_objective': objective["total"],
                    'objective_details': objective 
                    }
                )

        run = self.api.run(wandb.run.path)
        #run.summary["objective"] = result.get_objective()
        run.summary["hpo_configs"] = result.configs

        run.summary.update()

        # FIRESTORE
        if self.firestore_resolution > 0:
            # save the objective data into the experiment
            hpo_step_experiment_ref = self.db.collection("experiments").document(result.id)

            hpo_step_experiment_ref.update(
                {
                    "hpo_objective": result.get_objective(),
                    "parent_sweep": parent
                }
            )

            # mark this HPO step's experiment as belonging to the parent sweep
            parent_sweep_ref = self.db.collection("hpo_runs").document(parent)

            parent_sweep_ref.update(
                {
                    "runs": firestore.ArrayUnion([result.id])
                }
            )

    def create_hpo_sweep(self, sweep: HPOResult):
        """
        Creates an entry in Firestore for the given HPO sweep.

        Stores basic metadata that Weights & Biases tracks automatically
        but needs to be manually created in Firestore for other
        storage functions (such as save_hpo_step()) to work correctly.

        Args:
            sweep (HPOResult): the sweep for which to create an entry
        """
        if self.firestore_resolution > 0:

            # create the DB entry for this experiment in Firestore
            sweep_ref = self.db.collection("hpo_runs").document(sweep.id)

            sweep_ref.set(
                {
                    "name": sweep.name,
                    "start_time": sweep.start_time,
                    "best_run_id": None,
                    "runs": [],
                    "completed": False,
                    "configs": {
                        conf_name:json.dumps(conf_json)
                        for conf_name, conf_json in sweep.configs.items()
                    }
                }
            )


    def save_hpo_result(self, result: HPOResult):
        """
        Saves the final results of an HPO run to Firestore and
        marks it as completed.

        Includes end times, best run ID, and an ordered list of runs
        by objective value.

        Does not save the individual steps--you need to use
        save_hpo_step() for that.

        Args:
            result (HPOResult): the results of the completed HPO sweep to
            summarize and save
        """

        # WEIGHTS & BIASES automatically tracks this already

        # FIRESTORE
        # set the end time and best run ID and mark as completed
        if self.firestore_resolution > 0:

            sweep_ref = self.db.collection("hpo_runs").document(result.id)

            sweep_ref.update(
                {
                    "end_time": result.end_time,
                    "best_run_id": result.best_run.id,
                    "completed": True,
                    "runs_by_objective": [
                        step.id for step in result.get_top_k_steps(len(result.runs))
                        ]
                }
            )

    def create_artifact(self, content: dict) -> wandb.Artifact:
        """
        Creates a W&B artifact for saving in the database.

        Currently unused.

        Args:
            content (dict): the data to encapsulate in the Artifact
        Returns:
            wandb.Artifact: the encapsulated data
        """
        # Implementation to create and save a wandb.Artifact
        pass

    def average_nested_data(self, data):
        """
        Averages an arbitrarily deep data structure
        and returns the result as a single value.

        Used here to reduce the granularity of data in order
        to store a single value for each step in W&B.

        Args:
            data: the value(s) to reduce
        Returns:
            a single value representing the averaged data
        """
        if isinstance(data, list):
            if len(data) == 0:
                data=[0]
            ret = np.mean(np.nan_to_num([self.average_nested_data(item) for item in data]))
        elif hasattr(data, 'tolist'):  # numpy array
            if len(data) == 0:
                data=[0]
            ret = np.mean(np.nan_to_num(data))
        else:
            # Scalar value
            ret = data

        return ret.item() if isinstance(ret, np.generic) else ret
