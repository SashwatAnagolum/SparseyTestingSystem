from datetime import datetime
from sparseypy.core.results.result import Result
from sparseypy.core.results.hpo_step_result import HPOStepResult


class HPOResult(Result):
    """
    Hyperparameter Optimization Result: class to store the results of a hyperparameter optimization run.
    Attributes:
        name (str): The name of the hyperparameter optimization run.
        id (str): The id of the hyperparameter optimization run.
        best_run (HPOStepResult): The best run from the hyperparameter optimization.
        runs (list[HPOStepResult]): The list of runs from the hyperparameter optimization.
        configs (dict): The configurations for the hyperparameter optimization.
    """
    def __init__(self, configs: dict, id: str, name: str):
        """
        Initializes the HPOResult.
        Args:
            configs (dict): The configurations for the hyperparameter optimization.
            id (str): The id of the hyperparameter optimization run.
            name (str): The name of the hyperparameter optimization run.
        """
        super().__init__(configs)
        self.name = name
        self.id = id
        self.best_run = None
        self.runs = []  # List of HPOStepResult objects


    def add_step(self, step: HPOStepResult):
        """
        Add a step to the hyperparameter optimization result.
        Args:
            step (HPOStepResult): The step to add.
        """
        if self.best_run is None or step.get_objective()["total"] > self.best_run.get_objective()["total"]:
            self.best_run = step
        self.runs.append(step)

    def get_steps(self) -> list[HPOStepResult]:
        """
        Get the steps from the hyperparameter optimization result.
        Returns:
            (list[HPOStepResult]): The steps from the hyperparameter optimization result.
        """
        return self.runs

    def get_top_k_steps(self, k: int) -> list[HPOStepResult]:
        """
        Get the top k steps from the hyperparameter optimization result.
        Args:
            k (int): The number of top steps to return.
        Returns:
            (list[HPOStepResult]): The top k steps from the hyperparameter optimization result.
        """
        # sort copy by objective total, return top k
        return sorted(self.runs, key = lambda x : x.get_objective()["total"], reverse=True)[:k]