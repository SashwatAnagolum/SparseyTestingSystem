from datetime import datetime
from typing import Dict
from src.sparsepy.core.results import Result

class TrainingStepResult(Result):
    def __init__(self, id: str, start_time: datetime, end_time: datetime, resolution: str, metrics: Dict):
        super().__init__(id, start_time, end_time)
        self.resolution = resolution
        self.metrics = metrics

    def populate(self, metrics: Dict):
        pass
