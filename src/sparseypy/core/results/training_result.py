from typing import Optional

from sparseypy.core.results.result import Result
from sparseypy.core.results.training_step_result import TrainingStepResult
from sparseypy.core.metrics.metrics import Metric

class TrainingResult(Result):
    """
    Training Result: class to store the results of a training run.
    Attributes:
        id (str): The id of the training run.
        result_type (str): The type of result.
        results (list[TrainingStepResult]): The list of results from the training run.
        best_steps (dict): The best steps from the training run.
        configs (dict): The configurations for the training run.
    """
    def __init__(self, id: str, result_type: str, metrics: list[Metric], configs: Optional[dict] = None):
        """
        Initializes the TrainingResult.
        Args:
            id (str): The id of the training run.
            result_type (str): The type of result.
            metrics (list[Metric]): The metrics to track.
            configs (dict): The configurations for the training run.
        """
        super().__init__()
        self.id = id
        self.result_type = result_type
        self.results = []  # List of TrainingStepResult objects
        self.best_steps = {}
        self.configs = configs if configs else {}

        # get the best_item functions
        self.best_steps = {}
        for metric in metrics:
            self.best_steps[metric.get_name()] = {
                'best_index': 0,
                'best_value': None,
                'best_function': metric.get_best_comparison_function()
            }


    def add_step(self, step: TrainingStepResult):
        """
        Add a step to the training result.

        Args:
            step (TrainingStepResult): The step to add.
        """
        # add this step
        self.results.append(step)
        # and update the best values
        step_metrics = step.get_metrics()
        # for each metric we are tracking
        for metric_name, best_data in self.best_steps.items():
            # if that metric has no best value OR
            # if running the "best_function" comparison retrieved from the Metric at construction time
            # tells us that this value is better than the best value
            if best_data["best_value"] is None or best_data["best_function"](step_metrics[metric_name], best_data["best_value"]):
                # then update the best value and index for this metric to be the current step
                best_data['best_index'] = len(self.results)
                best_data['best_value'] = step_metrics[metric_name]

    def get_best_step(self, metric: str) -> TrainingStepResult:
        """
        Get the best step for a given metric.
        Args:
            metric (str): The metric to get the best step for.
        Returns:
            (TrainingStepResult): The best step for the given metric.
        """
        return self.results[self.best_steps[metric]["best_index"]]

    def get_step(self, index: int) -> TrainingStepResult:
        """
        Get a step by index.
        Args:
            index (int): The index of the step to get.
        Returns:
            (TrainingStepResult): The step at the given index.
        """
        return self.results[index]

    def get_steps(self) -> list[TrainingStepResult]:
        """ 
        Get the steps from the training result.
        Returns:
            (list[TrainingStepResult]): The steps from the training result.
        """
        return self.results

    def add_config(self, name, config):
        """
        Add a configuration to the training result.
        Args:
            name (str): The name of the configuration.
            config: The configuration to add.
        """
        self.configs[name] = config

    def get_configs(self) -> dict:
        """
        Get the configurations for the training run.
        Returns:
            (dict): The configurations for the training run.
        """
        return self.configs