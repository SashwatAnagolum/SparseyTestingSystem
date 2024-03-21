from firebase_admin import firestore
import json
import numpy as np
import pickle
import wandb

from sparsepy.core.results.training_result import TrainingResult
from sparsepy.core.results.training_result import TrainingStepResult
from sparsepy.core.results.hpo_result import HPOResult
from sparsepy.core.results.hpo_step_result import HPOStepResult
from sparsepy.access_objects.models.model import Model

"""
DataStorer: Stores data to weights and biases
"""
class DataStorer:

    def __init__(self, config: dict, database_resolution="none"):
        # configure saved metrics?
        self.saved_metrics = [metric["name"] for metric in config if metric["save"] is True]
        # create API client
        self.api = wandb.Api()

        # connect to Firestore
        self.db = firestore.client()

        self.database_resolution = database_resolution
        
    
    def save_model(self, m: Model):
        # Implementation to save the model
        pass
    
    def save_training_step(self, parent: str, result: TrainingStepResult):

        summary_dict = {}
        full_dict = {}
        
        # gather the saved metrics for summary in W&B and full storage in the DB
        for metric_name, metric_val in result.get_metrics().items():
            if metric_name in self.saved_metrics:
                summary_dict[metric_name] = self.average_nested_data(metric_val)
                full_dict[metric_name] = pickle.dumps(metric_val) # pickling
                #wandb.log({metric_name: self.average_nested_data(metric_val)}, commit=False)
        
        # save the summary to W&B
        # add this if we add step resolution back now that we don't need W&B concordance
        #summary_dict["resolution"] = result.resolution
        wandb.log(summary_dict)

        if self.database_resolution == "full":

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
        # gather the saved metrics for summary in W&B and full storage in the DB
        full_dict = {}
        for metric_name, metric_val in result.get_metrics().items():
            if metric_name in self.saved_metrics:
                full_dict[metric_name] = pickle.dumps(metric_val) # pickling

        if self.database_resolution == "full":

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
        
        if self.database_resolution != "none":
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
        # Implementation to save the training result

        # do we even need to set anything in W&B here? time finished? but W&B should track that
        # is there something we need to do here to mark end of run?
        # FIXME save required data if any, else remove
        run = self.api.run(wandb.run.path)

        if self.database_resolution != "none":

            experiment_ref = self.db.collection("experiments").document(result.id)

            # every invocation of this costs 1 read call; consider removing
            if experiment_ref.get().exists:
                experiment_ref.update(
                    {
                        "start_times": {
                            "training": result.start_time
                        },
                        "end_times": {
                            "training": result.end_time
                            },
                        "completed": True
                    }
                )

        # COMMENT THIS IN to test updated config saving
        #run.config = result.

    def save_evaluation_result(self, result: TrainingResult):
        # Implementation to save the evaluation result

        # access the current run's summary-level data with the API
        run = self.api.run(wandb.run.path)
        
        if self.database_resolution != "none":
        
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
                        "start_times": {
                            "evaluation": result.start_time
                        },
                        "end_times": {
                            "evaluation": result.end_time,
                            "experiment": result.end_time
                        },
                        "completed": True
                    }
                )

    def save_hpo_step(self, parent: str, result: HPOStepResult):
        # Implementation to save the HPO step result

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
        if self.database_resolution != "none":
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
        if self.database_resolution != "none":
        
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
                        conf_name:json.dumps(conf_json) for conf_name, conf_json in sweep.configs.items()
                    }
                }
            )


    def save_hpo_result(self, result: HPOResult):
        # Implementation to save the HPO result

        # WEIGHTS & BIASES automatically tracks this already

        # FIRESTORE
        # set the end time and best run ID and mark as completed
        if self.database_resolution != "none":
        
            sweep_ref = self.db.collection("hpo_runs").document(result.id)

            sweep_ref.update(
                {
                    "end_time": result.end_time,
                    "best_run_id": result.best_run.id,
                    "completed": True
                }
            )
    
    def create_artifact(self, content: dict) -> wandb.Artifact:
        # Implementation to create and save a wandb.Artifact
        pass

    def average_nested_data(self, data):
        if isinstance(data, list):
            if data.__len__() == 0:
                data=[0]
            ret = np.mean(np.nan_to_num([self.average_nested_data(item) for item in data]))
        elif hasattr(data, 'tolist'):  # numpy array
            if data.__len__() == 0:
                data=[0]
            ret = np.mean(np.nan_to_num(data))
        else:
            # Scalar value
            ret = data

        return ret.item() if isinstance(ret, np.generic) else ret
