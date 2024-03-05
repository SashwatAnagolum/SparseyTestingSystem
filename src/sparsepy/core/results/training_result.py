from datetime import datetime
from typing import Dict, List
from src.sparsepy.core.results import Result
from src.sparsepy.core.results import TrainingStepResult

class TrainingResult(Result):
    def __init__(self, id: str, start_time: datetime, end_time: datetime, resolution: str):
        super().__init__(id, start_time, end_time)
        self.resolution = resolution
        self.results: List[TrainingStepResult] = []
        self.final_val = TrainingStepResult()
        self.final_train = TrainingStepResult()
        self.best_validation_step = TrainingStepResult()
        self.best_training_step = TrainingStepResult()
        self.best_steps = {}

    def add_step(self, step: TrainingStepResult):
        pass

    def mark_finished(self):
        pass

    def get_best_step(self) -> TrainingStepResult:
        pass

