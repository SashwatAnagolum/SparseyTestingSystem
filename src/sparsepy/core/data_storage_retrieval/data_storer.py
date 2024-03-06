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

    def save_model(self, m: Model):
        # Implementation to save the model
        pass
    
    def save_training_step(self, parent: str, result: TrainingStepResult):
        # Implementation to save the training step result
        pass
    
    def save_training_result(self, result: TrainingResult):
        # Implementation to save the training result
        pass
    
    def save_evaluation_result(self, result: EvaluationResult):
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
