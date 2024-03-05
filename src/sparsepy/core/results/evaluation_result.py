from datetime import datetime
from typing import Dict
from src.sparsepy.core.results import Result

class EvaluationResult(Result):
    def __init__(self, id: str, start_time: datetime, end_time: datetime, dataset: str):
        super().__init__(id, start_time, end_time)
        self.dataset = dataset
        self.metrics = {}

    def populate(self, metrics: Dict):
        self.metrics = metrics
