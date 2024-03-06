from src.sparsepy.core.results import Result, TrainingResult, EvaluationResult
class HPOStepResult(Result):
    def __init__(self, parent_run: str, id: str, configs: dict):
        super().__init__()
        self.id = id
        self.parent_run = parent_run
        self.configs = configs
        self.training_results = None
        self.eval_results = None
        self.objective = None

    def populate(self, objective: dict, training_results: TrainingResult, eval_results: EvaluationResult):
        self.objective = objective
        self.training_results = training_results
        self.eval_results = eval_results

    def get_training_results(self) -> TrainingResult:
        return self.training_results

    def get_eval_results(self) -> EvaluationResult:
        return self.eval_results
    
    def get_objective(self) -> dict:
        return self.objective