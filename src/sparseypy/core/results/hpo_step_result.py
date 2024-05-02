from sparseypy.core.results.result import Result
from sparseypy.core.results.training_result import TrainingResult


class HPOStepResult(Result):
    """
    Hyperparameter Optimization Step Result: class to store the results of a single hyperparameter optimization step.
    Attributes:
        id (str): The id of the hyperparameter optimization step.
        parent_run (str): The id of the parent hyperparameter optimization run.
        configs (dict): The configurations for the hyperparameter optimization step.
        training_results (TrainingResult): The training results for the hyperparameter optimization step.
        eval_results (TrainingResult): The evaluation results for the hyperparameter optimization step.
        objective (dict): The objective values for the hyperparameter optimization step.
    """
    def __init__(self, parent_run: str, id: str, configs: dict):
        """
        Initializes the HPOStepResult.
        Args:
            parent_run (str): The id of the parent hyperparameter optimization run.
            id (str): The id of the hyperparameter optimization step.
            configs (dict): The configurations for the hyperparameter optimization step.
        """
        super().__init__(configs)
        self.id = id
        self.parent_run = parent_run
        self.training_results = None
        self.eval_results = None
        self.objective = None


    def populate(self, objective: dict, training_results: TrainingResult, eval_results: TrainingResult):
        """
        Populate the HPOStepResult with results.

        Args:
            objective (dict): The objective values for the hyperparameter optimization step.
            training_results (TrainingResult): The training results for the hyperparameter optimization step.
            eval_results (TrainingResult): The evaluation results for the hyperparameter optimization step.
        """
        self.objective = objective
        self.training_results = training_results
        self.eval_results = eval_results
        self.mark_finished()


    def get_training_results(self) -> TrainingResult:
        """
        Get the training results for the hyperparameter optimization step.

        Returns:
            (TrainingResult): The training results for the hyperparameter optimization step.
        """
        return self.training_results


    def get_eval_results(self) -> TrainingResult:
        """
        Get the evaluation results for the hyperparameter optimization step.

        Returns:
            (TrainingResult): The evaluation results for the hyperparameter optimization step.
        """
        return self.eval_results
    

    def get_objective(self) -> dict:
        """
        Get the objective values for the hyperparameter optimization step.

        Returns:
            (dict): The objective value for the hyperparameter optimization step.
        """
        return self.objective
