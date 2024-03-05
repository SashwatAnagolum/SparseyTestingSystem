from datetime import datetime
from src.sparsepy.core.results import Result, TrainingStepResult

class TrainingResult(Result):
    def __init__(self, id: str, resolution: str):
        super().__init__()
        self.id = id
        self.resolution = resolution
        self.results = []  # List of TrainingStepResult objects
        self.best_steps = {}

    def add_step(self, step: TrainingStepResult):
        self.results.append(step)

    def get_best_step(self, metric: str) -> TrainingStepResult:
        # Implementation needed to return the best step based on the metric
        pass

    def get_step(self, index: int) -> TrainingStepResult:
        return self.results[index]

    def get_steps(self) -> list[TrainingStepResult]:
        return self.results
