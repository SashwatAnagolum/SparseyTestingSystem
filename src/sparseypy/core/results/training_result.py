from typing import Optional

import torch

from sparseypy.core.results.result import Result
from sparseypy.core.results.training_step_result import TrainingStepResult
from sparseypy.core.metrics.metrics import Metric


class TrainingResult(Result):
    """
    TrainingResult: class to store the results of a training run.

    Attributes:
        id (str): the ID of this training run (in Weights & Biases)
        result_type (str): the type of this result (training/validation/evaluation)
        results (list[TrainingStepResult]): the results for each batch of this training run
        batch_start_indices (list[int]): the index of the first result in each batch/step 
            within the TrainingResult
        num_items (int): the current number of items in this TrainingResult
        best_steps (dict): the best step from this training run for each metric
        configs (dict): the config files for this training run
    """
    def __init__(self, id: str, result_type: str, metrics: list[Metric],
                 max_batch_size: int = 1, configs: Optional[dict] = None):
        """
        Initializes the TrainingResult.

        Args:
            id (str): The id of the training run.
            result_type (str): The type of result.
            metrics (list[Metric]): The metrics to track.
            max_batch_size (int): The maximum batch size stored in this TrainingResult.
            configs (dict): The configurations for the training run.
        """
        super().__init__(configs)
        self.id = id
        self.result_type = result_type
        self.results = []  # List of TrainingStepResult objects
        self.batch_start_indices = []
        self.best_steps = {}
        self.max_batch_size = max_batch_size
        self.num_items = 0
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
        # update the batch breakpoints
        self.batch_start_indices.append(self.num_items)
        # and update the best values and item counts
        step_metrics = step.get_metrics()
        for metric_name, best_data in self.best_steps.items():
            # for each item in this batch
            for batch_index in range(step.batch_size):
                # slice out the value of the metric for this batch item
                step_value = torch.select(step_metrics[metric_name], dim=1, index=batch_index)
                # if the metric has no best value OR
                # if running the "best_function" comparison retrieved from the Metric
                # at construction time tells us that this value is better than the best value
                if best_data["best_value"] is None or best_data["best_function"](
                    step_value, best_data["best_value"]
                ):
                    # then update the best value and index for this metric to the current item
                    best_data['best_batch'] = len(self.results) - 1
                    best_data['in_batch_index'] = batch_index
                    best_data['best_index'] = self.num_items + (batch_index + 1)
                    best_data['best_value'] = step_value
        # finally, update item count to account for the new batch
        self.num_items += step.batch_size


    def get_best_step(self, metric: str) -> TrainingStepResult:
        """
        Get the best step for a given metric.

        Args:
            metric (str): The metric to get the best step for.

        Returns:
            (TrainingStepResult): the batch of results containing the best step
            (int): the index of the best step within those results
        """
        return self.results[self.best_steps[metric]["best_batch"]], self.best_steps[metric]["in_batch_index"]


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
            (list[TrainingStepResult]): The steps
                from the training result.
        """
        return self.results
