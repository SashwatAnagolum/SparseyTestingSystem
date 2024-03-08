import numpy as np
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
    
    def save_model(self, m: Model):
        # Implementation to save the model
        pass
    
    def save_training_step(self, parent: str, result: TrainingStepResult):
        # Implementation to save the training step result
        #for metric_name, metric_vals in result.get_metrics():
        #    if metric_name in DataStorer.saved_metrics:
        #        wandb.log()
        # gather data to log
        logged_data = {
            
        }
        
        logged_data = {
            'resolution': result.resolution,
        }

        #wandb.log({k:self.average_nested_data(v) for k, v in result.get_metrics() if k in self.saved_metrics}, commit=False)

        #wandb.log(result.resolution, commit=True)

        pass
    
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