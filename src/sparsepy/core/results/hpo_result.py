from datetime import datetime
from datetime import datetime
from typing import Dict, List
from src.sparsepy.core.results import Result
from src.sparsepy.core.results import HPOStepResult

class HPOResult(Result):
    def __init__(self, id: str, start_time: datetime, end_time: datetime, name: str, configs: Dict):
        super().__init__(id, start_time, end_time)
        self.name = name
        self.best_run_id = None
        self.runs: List[HPOStepResult] = []
        self.configs = configs

    def add_step(self, step: HPOStepResult):
        pass

    def mark_finished(self):
        pass

    def get_best_run(self) -> HPOStepResult:
        pass
