from datetime import datetime

from sparsepy.core.results.result import Result

class TrainingStepResult(Result):
    def __init__(self, resolution: str):
        super().__init__()
        self.resolution = resolution
        self.metrics = []

    def add_metric(self, name: str, values: list):
        self.metrics[-1][name] = values

    def get_metric(self, batch: int, name: str) -> list:
        return self.metrics[batch].get(name, None)

    def get_metrics(self) -> dict:
        return self.metrics

    def add_batch(self):
        self.metrics.append({})