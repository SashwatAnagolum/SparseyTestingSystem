from firebase_admin import firestore
import json
import numpy as np
import pickle
import wandb

from sparsepy.core.results.training_result import TrainingResult
from sparsepy.core.results.training_result import TrainingStepResult
from sparsepy.core.results.evaluation_result import EvaluationResult
from sparsepy.core.results.hpo_result import HPOResult
from sparsepy.core.results.hpo_step_result import HPOStepResult
from sparsepy.access_objects.models.model import Model

"""
DataStorer: Stores data to weights and biases
"""
class DataStorer:

    def __init__(self, config: dict):
        # configure saved metrics?
        self.saved_metrics = [metric["name"] for metric in config if metric["save"] is True]
        # create API client
        self.api = wandb.Api()

        # connect to Firestore
        self.db = firestore.client()
        
    
    def save_model(self, m: Model):
        # Implementation to save the model
        pass
    
    def save_training_step(self, parent: str, result: TrainingStepResult):

        summary_dict = {}
        full_dict = {}
        
        # gather the saved metrics for summary in W&B and full storage in the DB
        for metric_name, metric_val in result.get_metrics()[0].items():
            if metric_name in self.saved_metrics:
                summary_dict[metric_name] = self.average_nested_data(metric_val)
                full_dict[metric_name] = pickle.dumps(metric_val) # pickling
                #wandb.log({metric_name: self.average_nested_data(metric_val)}, commit=False)
        
        # save the summary to W&B
        # add this if we add step resolution back now that we don't need W&B concordance
        #summary_dict["resolution"] = result.resolution
        wandb.log(summary_dict)

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
    
    def create_experiment(self, experiment: TrainingResult):
        # create the DB entry for this experiment in Firestore
        experiment_ref = self.db.collection("experiments").document(experiment.id)

        experiment_ref.set(
            {
                "start_time": experiment.start_time,
                "saved_metrics": {
                    "resolution": experiment.resolution
                },
                "completed": False
            }
        )

    def save_training_result(self, result: TrainingResult):
        # Implementation to save the training result
    def save_evaluation_result(self, parent: str, result: EvaluationResult):
        # Implementation to save the evaluation result
        pass
    
    def save_hpo_step(self, parent: str, result: HPOStepResult):
        # Implementation to save the HPO step result
        pass
    
    def save_hpo_result(self, result: HPOResult):
        # Implementation to save the HPO result
        pass
    
    def create_artifact(self, content: dict) -> wandb.Artifact:
        # Implementation to create and save a wandb.Artifact
        pass

    def set_saved_metrics(self, metrics: list[str]):
        DataStorer.saved_metrics = metrics

    def average_nested_data(self, data):
        if isinstance(data, list):
            return np.mean(np.nan_to_num([self.average_nested_data(item) for item in data]))
        elif hasattr(data, 'tolist'):  # numpy array
            return np.mean(np.nan_to_num(data))
        else:
            # Scalar value
            return data