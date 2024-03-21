from datetime import datetime
from sparsepy.core.results.result import Result
from sparsepy.core.results.hpo_step_result import HPOStepResult

class HPOResult(Result):
    def __init__(self, configs: dict, id: str, name: str):
        super().__init__()
        self.name = name
        self.id = id
        self.best_run = None
        self.runs = []  # List of HPOStepResult objects
        self.configs = configs

    def add_step(self, step: HPOStepResult):
        if self.best_run is None or step.get_objective()["total"] > self.best_run.get_objective()["total"]:
            self.best_run = step
        self.runs.append(step)

    def get_steps(self) -> list[HPOStepResult]:
        return self.runs

    def get_top_k_steps(self, k: int) -> list[HPOStepResult]:
        # sort copy by objective total, return top k
        return sorted(self.runs, key = lambda x : x.get_objective()["total"], reverse=True)[:k]