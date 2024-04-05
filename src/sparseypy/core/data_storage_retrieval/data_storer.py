# -*- coding: utf-8 -*-
"""
DataStorer: Saves data to Weights & Biases and the system database (Firestore)
"""
from datetime import datetime, timezone
import json
import pickle

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
        # ensure the config has been initialized
        if not DataStorer.is_initialized:
            raise ValueError("You must call DataStorer.configure() before intializing DataStorer objects.")

        # configure saved metrics?
        self.saved_metrics = [metric["name"] for metric in metric_config if metric["save"] is True]
        # create API client
        self.api = wandb.Api()

        # connect to Firestore
        self.db = firestore.client()

        self.wandb_resolution = DataStorer.wandb_config["data_resolution"]
        self.firestore_resolution = DataStorer.firestore_config["data_resolution"]

        self.step_cache = {
            "training": [],
            "evaluation": []
        }

        self.tables = DataStorer.firestore_config["table_names"]

        self.batch_size = DataStorer.firestore_config["batch_size"]

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

    def save_model(self, experiment: str, m: Model, model_config: dict):
        """
        Saves a model to Weights & Biases.

        Args:
            experiment (str): the experiment ID to which the model should be saved
            m (Model): the model object to be saved
        """
        if DataStorer.wandb_config["save_models"]:
            # WEIGHTS & BIASES
            # this section creates the artifact manually because doing so provides
            # more flexibility than using link_model()
            # instance name: the unique name of this specific model version
            instance_name = experiment + "-model"
            # model name: the user-defined model name that this model is a version of
            model_name = model_config["model_name"] if model_config["model_name"] else instance_name
            # model description: string description to log into W&B model registry
            model_description = model_config["model_description"]
            if model_description is None:
                model_description = f"Automatically generated by run {experiment}."

            # create the model artifact
            model_artifact = wandb.Artifact(
                name=instance_name,
                type="model",
                description=model_description
            )

            # add the state dict
            with model_artifact.new_file("model.pt", mode="wb") as f:
                torch.save(m.state_dict(), f)

            # add the model config file (required for rehydration)
            with model_artifact.new_file("network.yaml", encoding="utf-8") as f:
                json.dump(model_config, f)

            # log to W&B and link to the correct model registry entry
            wandb.log_artifact(model_artifact).wait()
            wandb.run.link_artifact(model_artifact, f"model-registry/{model_name}")

            # FIRESTORE
            if self.firestore_resolution > 0:
                # update "model registry" table (named versions of models)
                reg_ref = self.db.collection(self.tables["model_registry"]).document(model_name)

                if reg_ref.get().exists:
                    reg_ref.update(
                        {
                            "last_updated": datetime.now(),
                            "versions": firestore.ArrayUnion([instance_name])
                        }
                    )
                else:
                    reg_ref.set(
                        {
                            "last_updated": datetime.now(),
                            "versions": [
                                    {
                                        "id": instance_name,
                                        "wandb_location": model_artifact.source_qualified_name
                                    }
                                ]
                        }
                    )

                # update "models" table (individual model instances/versions)
                model_ref = self.db.collection(self.tables["models"]).document(instance_name)

                model_entry = {
                        'registry_name': model_name,
                        'config': model_config,
                        'wandb_location': model_artifact.source_qualified_name
                    }

                if DataStorer.firestore_config["save_models"]:
                    model_entry['trained_weights'] = pickle.dumps(m.state_dict()),

                model_ref.set(
                    model_entry
                )

    def _flush_cache(self, experiment: str, phase=None):
        """
        Flushes the step cache to the database to ensure read consistency.

        You need to call this method under two circumstances:
        1) as part of saving the final results for a phase to ensure all steps are written 
            to the database
        2) if you are reading step data back from the database before a phase is completed

        Args:
            experiment (str): the experiment to which the cache items should be saved.
            phase (str): the phase to flush the cache for. If this is not provided, all
                phases will be flushed.
        """
        if phase is not None:
            if len(self.step_cache[phase]) > 0:
                self._save_batch(experiment, phase)
        else:
            for phase, phase_data in self.step_cache.items():
                if len(phase_data) > 0:
                    self._save_batch(experiment, phase)

    def _save_batch(self, experiment: str, phase: str):
        """
        Saves a batch of training/evaluation steps to the database.

        Args:
            experiment (str): the experiment ID to save the batch to
            phase (str): the phase (training/evaluation) to save the batch for.
        """
        batch_id = str(datetime.now(timezone.utc).timestamp())

        batch_ref = self.db.collection(self.tables["batches"]).document(batch_id)

        this_batch_size = len(self.step_cache[phase])

        batch_ref.set(
            {
                "batch_type": phase,
                "steps": self.step_cache[phase],
                "size": this_batch_size,
                "parent": experiment
            }
        )

        experiment_ref = self.db.collection(self.tables["experiments"]).document(experiment)

        experiment_ref.update(
            {
                f"saved_metrics.{phase}": firestore.ArrayUnion(
                    [
                        {
                            "batch": batch_id,
                            "size": this_batch_size
                        }
                    ]
                )
            }
        )

        # clear the cache
        self.step_cache[phase] = []

    def _save_firestore_step(self, experiment: str, phase: str, metric_data: dict):
        """
        Save a single training/evaluation step to Firestore.

        Args:
            phase (str): the phase of training (training/evaluation) this step is for.
            parent (str): the experiment to which the step belongs.
        """
        if len(self.step_cache[phase]) >= self.batch_size:
            self._save_batch(experiment, phase)

        self.step_cache[phase].append(metric_data)

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
            for met_name, met_val in result.get_metrics().items():
                # if that metric is requested for saving and is at least 1D in layers (is a list)
                if met_name in self.saved_metrics and isinstance(met_val, (list, torch.Tensor)):
                    # then break out each layer as a separate metric for W&B using prefix grouping
                    for idx, layer_data in enumerate(met_val):
                        layer_name = f"{met_name}/layer_{idx}"
                        layerwise_dict[layer_name] = self.average_nested_data(layer_data)
                        # if the resolution is 2 (MAC-level)
                        # also log the MAC-level data with prefix grouping
                        # FIXME tensors are not logged here pending adjustment of feature_coverage
                        if self.wandb_resolution == 2 and isinstance(layer_data, list):
                            for idy, mac_data in enumerate(layer_data):
                                mac_name = f"{met_name}/layer_{idx}/mac_{idy}"
                                layerwise_dict[mac_name] = self.average_nested_data(mac_data)
            # then log without updating the step (done when the summary is logged below)
            wandb.log(layerwise_dict, commit=False)

        # save the summary to W&B
        # add this if we add step resolution back now that we don't need W&B concordance
        #summary_dict["resolution"] = result.resolution
        wandb.log(summary_dict)

        # save to Firestore on "step" resolution only
        if self.firestore_resolution == 2:
            self._save_firestore_step(parent, "training", full_dict)

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
            self._save_firestore_step(parent, "evaluation", full_dict)

    def create_experiment(self, experiment: TrainingResult):
        """
        Creates a new entry for the current experiment in Firestore.
        
        Args:
            experiment (TrainingResult): the TrainingResult for the new experiment
            for which to create a database entry
        """

        # log dataset description to W&B
        if experiment.configs:
            dataset_description = experiment.configs["dataset_config"]["description"]
            if dataset_description:
                wandb.run.summary["dataset_description"] = dataset_description

        # save on "summary" or better
        if self.firestore_resolution > 0:
            # create the DB entry for this experiment in Firestore
            experiment_ref = self.db.collection(self.tables["experiments"]).document(experiment.id)

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
                    "completed": False,
                    "batch_size": self.batch_size,
                    "dataset_description": dataset_description
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

        # all data is already tracked in W&B so we don't need to save anything special here

        # save on "summary" or better
        if self.firestore_resolution > 0:

            # flush the training result cache to the database
            self._flush_cache(experiment=result.id, phase="training")

            experiment_ref = self.db.collection(self.tables["experiments"]).document(result.id)

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
            wandb.run.summary["evaluation_" + metric_name] = metric_val

        if self.firestore_resolution > 0:
            # flush the evaluation cache
            self._flush_cache(experiment=result.id, phase="evaluation")
            # save the full results to Firestore
            experiment_ref = self.db.collection(self.tables["experiments"]).document(result.id)
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
            hpo_step_exp_ref = self.db.collection(self.tables["experiments"]).document(result.id)

            hpo_step_exp_ref.update(
                {
                    "hpo_objective": result.get_objective(),
                    "parent_sweep": parent
                }
            )

            # mark this HPO step's experiment as belonging to the parent sweep
            parent_sweep_ref = self.db.collection(self.tables["hpo_runs"]).document(parent)

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
            sweep_ref = self.db.collection(self.tables["hpo_runs"]).document(sweep.id)

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

            sweep_ref = self.db.collection(self.tables["hpo_runs"]).document(result.id)

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
