from datetime import datetime
from typing import Dict
from src.sparsepy.core.results import Result
from src.sparsepy.core.results import TrainingResult
from src.sparsepy.core.results import EvaluationResult

class HPOStepResult(Result):
    def __init__(self, id: str, start_time: datetime, end_time: datetime, parent_run: str, objective: Dict, configs: Dict):
        super().__init__(id, start_time, end_time)
        self.parent_run = parent_run
        self.objective = objective
        self.training_results = TrainingResult()
        self.eval_results = EvaluationResult()
        self.run_configs = configs
    def populate(self, objective: Dict, training_results: TrainingResult, eval_results: EvaluationResult):
        self.objective = objective
        self.training_results = training_results
        self.eval_results = eval_results