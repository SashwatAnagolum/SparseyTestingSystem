"""
firestore_db_adapter.py - database adapter class for the Firestore database
"""

from datetime import datetime, timezone
from functools import lru_cache
import json
import pickle

import firebase_admin
from firebase_admin import firestore # firestore_async for async client
from google.api_core.datetime_helpers import DatetimeWithNanoseconds
import torch

from sparseypy.access_objects.models.model import Model
from sparseypy.core.db_adapters.db_adapter import DbAdapter
from sparseypy.core.metrics import comparisons
from sparseypy.core.results import HPOResult, HPOStepResult, TrainingResult, TrainingStepResult


class FirestoreDbAdapter(DbAdapter):
    """
    FirestoreDbAdapter: database adapter for the Firestor
        database. Supports reading and writing.

    Key limitations: 1MiB maximum document size requires some
    care about how results are structured for large experiment
    runs.

    Attributes:
        config (dict): the adapter configuration information, inherited from superclass.
        is_initialized (bool): whether the adapter has been initialized and the client is logged in.
    """

    is_initialized = False

    def __init__(self, config: dict, metric_config: dict) -> None:
        """
        Initializes the FirestoreDbAdapter object. 

        Args:
            config (dict): the Firestore database adapter configuration
            metric_config (dict): the metric configuration, used to
                initialize the list of metrics to save.
        """
        super().__init__(config, metric_config)

        # only log in if we are not already logged in
        if not FirestoreDbAdapter.is_initialized:
            # initialize Firestore
            cred_obj = firebase_admin.credentials.Certificate(
                self.config['firebase_service_key_path']
            )

            firebase_admin.initialize_app(cred_obj)
            # mark as initialized
            FirestoreDbAdapter.is_initialized = True

        # create db client
        self.db = firestore.client()

        self.batch_size = self.config["batch_size"]
        self.resolution = self.config["data_resolution"]
        self.tables = self.config["table_names"]

        self.step_cache = {
            "training": [],
            "evaluation": []
        }


    def save_model(self, experiment: str, m: Model, model_config: dict, wandb_location: str):
        """
        Saves a model to Firestore.

        Args:
            experiment (str): the experiment ID to which the model should be saved
            m (Model): the model object to be saved
            model_config (dict): the model configuration to save with the model
                (required for rehydration)
            wandb_location (str): the qualified name/location of the model binary in
                Weights & Biases.
        """
        # instance name: the unique name of this specific model version
        instance_name = experiment + "-model"
        # model name: the user-defined model name that this model is a version of
        model_name = model_config["model_name"] if model_config["model_name"] else instance_name
        # model description: string description to log into W&B model registry
        model_description = model_config["model_description"]
        if model_description is None:
            model_description = f"Automatically generated by run {experiment}."

        # save to database
        if self.resolution > 0:
            # update "model registry" table (named versions of models)
            reg_ref = self.db.collection(self.tables["model_registry"]).document(model_name)

            if reg_ref.get().exists:
                reg_ref.update(
                    {
                        "last_updated": datetime.now(),
                        "source_run": experiment,
                        "versions": firestore.ArrayUnion([instance_name])
                    }
                )
            else:
                reg_ref.set(
                    {
                        "last_updated": datetime.now(),
                        "source_run": experiment,
                        "versions": [
                                {
                                    "id": instance_name,
                                    "wandb_location": wandb_location
                                }
                            ]
                    }
                )

            # update "models" table (individual model instances/versions)
            model_ref = self.db.collection(self.tables["models"]).document(instance_name)

            model_entry = {
                    'registry_name': model_name,
                    'config': model_config,
                    'wandb_location': wandb_location
                }

            if self.config["save_models"]:
                model_entry['trained_weights'] = pickle.dumps(m.state_dict())

            model_ref.set(
                model_entry
            )


    def create_experiment(self, experiment: TrainingResult):
        """
        Creates a new entry for the current experiment in Firestore.
        
        Args:
            experiment (TrainingResult): the TrainingResult for the new experiment
            for which to create a database entry
        """
        training_ds_config = experiment.get_config(
            'training_dataset_config'
        )

        eval_ds_config = experiment.get_config(
            'evaluation_dataset_config'
        )

        training_recipe_config = experiment.get_config(
            'training_recipe_config'
        )

        training_dataset_description = None
        evaluation_dataset_description = None
        training_recipe_description = None

        if training_ds_config is not None:
            training_dataset_description = training_ds_config.get(
                'description', None
            )

        if eval_ds_config is not None:
            evaluation_dataset_description = eval_ds_config.get(
                'description', None
            )

        if training_recipe_config is not None:
            training_recipe_description = training_recipe_config.get(
                'description'
            )

        # save on "summary" or better
        if self.resolution > 0:
            # create the DB entry for this experiment in Firestore
            experiment_ref = self.db.collection(self.tables["experiments"]).document(experiment.id)

            experiment_ref.set(
                {
                    "start_times": {
                        "experiment": experiment.start_time,
                        "training": experiment.start_time
                    },
                    "saved_metrics": {},
                    "end_times": {},
                    "completed": False,
                    "batch_size": self.batch_size,
                    "dataset_descriptions": 
                        {
                            "training": training_dataset_description,
                            "evaluation": evaluation_dataset_description
                        },
                    "description": training_recipe_description
                }
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
        experiment_data = self._get_experiment_data(experiment_id)

        metrics = []
        tr = TrainingResult(
            id=experiment_id,
            result_type=result_type,
            metrics=metrics,
            max_batch_size=1,
            configs={
                    conf_name:json.loads(conf_data)
                    for conf_name, conf_data in experiment_data["configs"]
                }
        )

        saved_metrics = experiment_data.get("saved_metrics", {}).get(result_type, [])
        for step_index in range(len(saved_metrics)):
            step_result = self.get_training_step_result(experiment_id, step_index, result_type)
            tr.add_step(step_result)

        tr.start_time = self._convert_firestore_timestamp(
                experiment_data.get("start_times", {}).get(result_type)
            )
        tr.end_time = self._convert_firestore_timestamp(
                experiment_data.get("end_times", {}).get(result_type)
            )

        step_dict = experiment_data.get("best_steps", {}).get(result_type, {})

        tr.best_steps = self._rehydrate_best_steps(step_dict)

        return tr


    def _rehydrate_best_steps(self, step_dict: dict) -> dict:
        """
        Reconstructs the best-performing steps for a TrainingResult from the dictionary value
        as saved in Firestore.

        Args:
            step_dict (dict): the serialized best steps retrieved from Firestore

        Returns:
            (dict): the rehydrated best steps for insertion into a TrainingResult
        """
        best_steps = {}

        for metric, metric_data in step_dict.items():
            best_function = metric_data.get("best_function")
            best_value_bytes = metric_data.get("best_value")

            # this takes a few liberties with the best steps to deal with batches
            # regardless of what the batch size in the DB was, the batch size for
            # a reloaded training result will be 1
            # so the best batch number will also be equal to the best index
            # and the in-batch index is always zero since all batches are size 1
            best_steps[metric] = {
                "best_batch": metric_data["best_index"],
                "best_function": getattr(comparisons, best_function),
                "best_index": metric_data["best_index"],
                "best_value": pickle.loads(best_value_bytes),
                "in_batch_index": 0
            }

        return best_steps



    def _unnest_tensor(self, values: torch.Tensor):
        """
        If the input is a NestedTensor, unbinds the values, converts to NumPy, moves to the CPU,
        and returns a list.

        Args:
            values (torch.Tensor): the input to unbind

        Returns:
            (list | torch.Tensor): the original tensor (if not a NestedTensor) 
                or the unbound values as a list
        """
        # only un-nest if the Tensor is nested
        if isinstance(values, torch.Tensor) and values.is_nested:
            # return the numpy representation for each of the Tensors x in the NestedTensor
            # to do this we unbind the NestedTensor
            # to do so we first need to copy it to contiguous memory
            # and move it to the CPU
            return [
                x.numpy() for x in values.contiguous().cpu().unbind()
            ]
            # if values.is_contiguous():
            #     return [
            #         x.numpy() for x in values.cpu().unbind()
            #     ]
            # else:
            #     return [
            #         x.cpu().numpy() for x in values.unbind()
            #     ]
        else:
            return values


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
        # save on "summary" or better
        if self.resolution > 0:

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
                        f"batch_sizes.{result.result_type}": result.max_batch_size,
                        "best_steps.training": self._extract_best_steps(result),
                        "configs": {
                            conf_name: json.dumps(conf_data)
                            for conf_name, conf_data in result.configs.items()
                        }
                    }
                )


    def _extract_best_steps(self, result: TrainingResult) -> dict:
        """
        Extract the best-performing steps from a TrainingResult into a dictionary for
        saving in Firestore.

        Args:
            result (TrainingResult): the result from which to extract the best steps

        Returns:
            (dict): the best steps in a serialized format suitable for saving to Firestore
        """
        return {
            metric_name:{
                'best_batch': metric_vals["best_batch"],
                'best_function': metric_vals["best_function"].__name__,
                'best_index': metric_vals["best_index"],
                'best_value': pickle.dumps(
                    self._unnest_tensor(metric_vals["best_value"]),
                ),
                'in_batch_index': metric_vals["in_batch_index"]
            }
            for metric_name, metric_vals in result.best_steps.items()
        }


    def get_training_step_result(
            self,
            experiment_id: str,
            step_index: int,
            result_type: str ="training"
        ) -> TrainingStepResult:
        """
        Retrieves the result of a specific training step within an experiment from Firestore.

        Args:
            experiment_id (str): The unique identifier for the experiment.
            step_index (int): The index of the training step to retrieve.
            result_type (str): The type of result to retrieve. Defaults to "training".

        Returns:
            TrainingStepResult: An instance of TrainingStepResult containing the step's metrics.

        Raises:
            ValueError: If the step index is out of bounds for the given experiment.
        """
        experiment_data = self._get_experiment_data(experiment_id)
        training_steps = experiment_data.get("saved_metrics", {}).get(result_type, [])
        if step_index < 0 or step_index >= len(training_steps):
            raise ValueError("Step index is out of bounds for the given experiment.")

        # fetch the step metadata (batch containing the step data and its index within the batch)
        batch_index = step_index // experiment_data["batch_size"] # integer division
        step_offset = step_index % experiment_data["batch_size"]

        # fetch the batch
        batch_data = self._get_batch_data(training_steps[batch_index]["batch"])
        # retrieve the step data from the batch using the index
        step_data = batch_data["steps"][step_offset]

        step_result = TrainingStepResult(batch_size=1)

        for metric_name, metric_data in step_data.items():
            step_result.add_metric(name=metric_name, values=pickle.loads(metric_data))

        return step_result


    def save_training_step(self, parent: str, result: TrainingStepResult,
                           phase: str = "training"):
        """
        Saves a single training step to Weights & Biases and Firestore.

        Args:
            parent (str): the experiment ID to which to log this step
            result (TrainingStepResult): the step results to save
            phase (str): the type of step to save (training/validation/evaluation)
        """
        if self.resolution == 2:
            # for each item in the batch in this TSR
            # (this batch size is the *step's* batch size, not the Firestore batch)
            # (and the two do not need to be the same)
            for batch_index in range(result.batch_size):
                # gather the saved metrics for full storage in the DB
                full_dict = {}
                # for each metric in the TSR
                for metric_name, metric_val in result.get_metrics().items():
                    # that is also on the list of saved metrics
                    if metric_name in self.saved_metrics:
                        # serialize the values from its result tensor...
                        full_dict[metric_name] = pickle.dumps(
                            # ... by first selecting only the entries corresponding
                            # to the current batch
                            # ... then un-nesting those values so we can pickle them
                            self._unnest_tensor(
                                torch.select(metric_val, dim=1, index=batch_index)
                            )
                        )
                # then save this batch item as a step to Firestore
                self._save_firestore_step(experiment=parent, phase=phase, metric_data=full_dict)


    def get_evaluation_result(self, experiment_id: str) -> TrainingResult:
        """
        Get the evaluation result for a given experiment from Firestore.

        Args:
            experiment_id (str): The ID of the experiment.

        Returns:
            EvaluationResult: the EvaluationResult for the experiment of this id in w&b
        """
        return self.get_training_result(experiment_id=experiment_id, result_type="evaluation")


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
        if self.resolution > 0:
            # flush the evaluation cache
            self._flush_cache(experiment=result.id, phase="evaluation")
            # save the full results to Firestore
            experiment_ref = self.db.collection(self.tables["experiments"]).document(result.id)
            if experiment_ref.get().exists:
                # add step to existing experiment
                experiment_ref.update(
                    {
                        f"batch_sizes.{result.result_type}": result.max_batch_size,
                        "best_steps.evaluation": self._extract_best_steps(result),
                        "completed": True,
                        "end_times.evaluation": result.end_time,
                        "end_times.experiment": result.end_time,
                        "start_times.evaluation": result.start_time,
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
        if self.resolution > 0:

            # create the DB entry for this experiment in Firestore
            sweep_ref = self.db.collection(self.tables["hpo_runs"]).document(sweep.id)

            description = sweep.configs["hpo_config"]["description"] if sweep.configs else None

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
                    },
                    "description": description
                }
            )


    def get_hpo_step_result(self, hpo_run_id, experiment_id):
        """
        Retrieves the result of a specific experiment step within an HPO run from Firestore.

        This method combines experiment data and HPO configuration to create a comprehensive
        step result for hpo.

        Args:
            hpo_run_id (str): The unique identifier for the HPO run.
            experiment_id (str): The unique identifier for the experiment within the HPO run.

        Returns:
            HPOStepResult: An instance of HPOStepResult representing the experiment step 
            within the HPO run.
        """
        experiment_data = self._get_experiment_data(experiment_id)
        training_result = self.get_training_result(experiment_id)
        evaluation_result = self.get_evaluation_result(experiment_id)
        hpo_run_data = self._get_hpo_run_data(hpo_run_id)
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


    def save_hpo_step(self, parent: str, result: HPOStepResult):
        """
        Saves a single HPO step to Firestore.

        Saves objective data and HPO configuration to the run in
        both Firestore.

        Also marks this experiment in Firestore as belonging to the
        parent sweep and updates its best runs.
        
        Args:
            parent (str): the ID of the parent sweep in the HPO table
            that should be updated with this run's results
            result (HPOStepResult): the results of the HPO step to save
        """
        if self.resolution > 0:
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


    def get_hpo_result(self, hpo_run_id: str) -> HPOResult:
        """
        Retrieves the overall result of a specific hyperparameter optimization (HPO) run
        from Firestore.

        This method aggregates the results of individual experiments within an HPO run, 
        and provides a comprehensive view of the HPO run, including start and end times, 
        configuration settings, and the best-performing experiment.

        Args:
            hpo_run_id (str): The unique identifier for the HPO run.

        Returns:
            HPOResult: An instance of HPOResult containing aggregated results 
            and configuration info from the HPO run.
        """
        hpo_run_data = self._get_hpo_run_data(hpo_run_id)

        configs = {
            conf_name: json.loads(conf_json)
            for conf_name, conf_json in hpo_run_data["configs"].items()
            }
        hpo_result = HPOResult(configs=configs, id=hpo_run_id, name=hpo_run_data["name"])

        for experiment_id in hpo_run_data["runs"]:
            step_result = self.get_hpo_step_result(hpo_run_id, experiment_id)
            hpo_result.add_step(step_result)

        hpo_result.best_run = self.get_hpo_step_result(hpo_run_id, hpo_run_data["best_run_id"])
        hpo_result.start_time = self._convert_firestore_timestamp(hpo_run_data["start_time"])
        hpo_result.end_time = self._convert_firestore_timestamp(hpo_run_data["end_time"])
        return hpo_result


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
        # set the end time and best run ID and mark as completed
        if self.resolution > 0:

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


    def _convert_firestore_timestamp(
            self,
            firestore_timestamp: DatetimeWithNanoseconds
        ) -> datetime:
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


    @lru_cache(maxsize=16)
    def _get_experiment_data(self, experiment_id) -> dict:
        """
        Retrieves and caches the data for a specific experiment from Firestore.

        Args:
            experiment_id (str): The unique identifier for the experiment.

        Returns:
            dict: A dictionary containing the experiment data.
        """
        experiment_ref = self.db.collection(self.tables["experiments"]).document(experiment_id)
        return experiment_ref.get().to_dict()


    @lru_cache(maxsize=16)
    def _get_batch_data(self, batch_id) -> dict:
        """
        Retrieves and caches the data for a specific batch of steps from Firestore.

        Args:
            batch_id (str): The unique identifier for the batch.

        Returns:
            dict: A dictionary containing the batch data.
        """
        batch_ref = self.db.collection(self.tables["batches"]).document(batch_id)
        return batch_ref.get().to_dict()


    @lru_cache(maxsize=16)
    def _get_hpo_run_data(self, hpo_run_id) -> dict:
        """
        Retrieves and caches the data for a specific HPO run from Firestore.

        Args:
            hpo_run_id (str): The unique identifier for the HPO run.

        Returns:
            dict: A dictionary containing the HPO run data.
        """
        hpo_run_ref = self.db.collection(self.tables["hpo_runs"]).document(hpo_run_id)
        return hpo_run_ref.get().to_dict()
