from typing import Optional

from sparseypy.core.results.result import Result
from sparseypy.core.results.training_step_result import TrainingStepResult
from sparseypy.core.metrics.metrics import Metric

class TrainingResult(Result):
    def __init__(self, id: str, result_type: str, resolution: str, metrics: list[Metric], configs: Optional[dict] = None):
        super().__init__()
        self.id = id
        self.resolution = resolution
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
        return self.results[self.best_steps[metric]["best_index"]]

    def get_step(self, index: int) -> TrainingStepResult:
        return self.results[index]

    def get_steps(self) -> list[TrainingStepResult]:
        return self.results

    def add_config(self, name, config):
        self.configs[name] = config

    def get_configs(self) -> dict:
        return self.configs