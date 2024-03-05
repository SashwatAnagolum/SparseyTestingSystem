from src.sparsepy.core.results import Result

class EvaluationResult(Result):
    def __init__(self, dataset: str):
        super().__init__()
        self.dataset = dataset
        self.metrics = {}

    def add_metric(self, name: str, values: list):
        self.metrics[name] = values

    def get_metric(self, name: str) -> list:
        return self.metrics.get(name, [])

    def get_metrics(self) -> dict:
        return self.metrics
