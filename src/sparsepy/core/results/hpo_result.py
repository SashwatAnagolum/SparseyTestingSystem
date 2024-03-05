from datetime import datetime
from src.sparsepy.core.results import Result, HPOStepResult

class HPOResult(Result):
    def __init__(self, configs: dict, id: str, name: str):
        self.name = name
        self.best_run_id = id
        self.runs = []  # List of HPOStepResult objects
        self.configs = configs

    def add_step(self, step: HPOStepResult):
        self.runs.append(step)

    def get_steps(self) -> list[HPOStepResult]:
        return self.runs

    def get_top_k_steps(self, k: int) -> list[HPOStepResult]:
        # Implementation needed to get top k steps