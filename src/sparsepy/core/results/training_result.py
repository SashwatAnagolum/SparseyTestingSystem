from datetime import datetime
from sparsepy.core.results.result import Result
from sparsepy.core.results.training_step_result import TrainingStepResult

class TrainingResult(Result):
    def __init__(self, id: str, resolution: str):
        super().__init__()
        self.id = id
        self.resolution = resolution
        self.results = []  # List of TrainingStepResult objects
        self.best_steps = {}

    def add_step(self, step: TrainingStepResult):
        # TODO update metrics to have some kind of average that can compare for best steps
        self.results.append(step)

    def get_best_step(self, metric: str) -> TrainingStepResult:
        # Implementation needed to return the best step based on the metric
        return self.best_steps.get(metric, None)

    def get_step(self, index: int) -> TrainingStepResult:
        return self.results[index]

    def get_steps(self) -> list[TrainingStepResult]:
        return self.results
