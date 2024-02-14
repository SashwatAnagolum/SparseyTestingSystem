from sparsepy.core.results import Result

class HPOResult(Result):
    def __init__(self, results, objective):
        
        super.__init__(results)
        self.objective = objective