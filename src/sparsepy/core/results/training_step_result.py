from datetime import datetime
from src.sparsepy.core.results import Result

class TrainingStepResult(Result):
    def __init__(self, resolution: str):
        super().__init__()
        self.resolution = resolution
        self.metrics = {}

    def add_metric(self, name: str, values: list):
        self.metrics[name] = values

    def get_metric(self, name: str) -> list:
        return self.metrics.get(name, None)

    def get_metrics(self) -> dict:
        return self.metrics

