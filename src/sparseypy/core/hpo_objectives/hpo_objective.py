"""
hpo_objective.py - contains the class to calculate HPO objective values
"""
import numpy as np

from sparseypy.core.metrics import comparisons
from sparseypy.core.results import TrainingResult

class HPOObjective:
    """
    Performs the calculations required to determine the value of the HPO objective function from
    the experiment results.

    Attributes:
        hpo_config (dict): the HPO configuration
    """
    def __init__(self, hpo_config: dict):
        """
        Constructor for the HPO Objective. Accepts the HPO configuration containing the
        required values to initialize the objective function.

        Args:
            hpo_config (dict): the validated system HPO configuration.
        """
        self.hpo_config = hpo_config


    def combine_metrics(self, results: TrainingResult) -> float:
        """
        Combines multiple metric results into a single scalar value using a specified 
        operation and weights, averaging values at different levels within each metric. 
        Only metrics specified in the HPO configuration are used.

        Args:
            results (TrainingResult): a TrainingResult object with the results of the experiment.
        Returns:
            (float): a single scalar value representing the combined result.
        """
        operation = self.hpo_config['optimization_objective']['combination_method']
        objective_terms = self.hpo_config['optimization_objective']['objective_terms']

        # set up the dictionary
        obj_vals = {
            'total': 0.0,
            'combination_method': self.hpo_config['optimization_objective']['combination_method'],
            'terms': {}
        }

        # for each metric in the objective
        for term in objective_terms:
            # get the correct format of the name
            metric_name = term["metric"]["name"]
            # for each result in the results get the averaged value of that metric into a list
            # REVIEW this since it will probably be broken by the TSR change
            term_values = [
                comparisons.average_nested_data(step.get_metric(metric_name))
                for step in results.get_steps()
            ]

            # average the values across all the steps to get the subtotal; also record the weight
            obj_vals["terms"][metric_name] = {'value': np.mean(term_values), 
                                                  'weight': term["weight"]}

        # weight all the values
        weighted_objectives = [
            term["value"] * term["weight"]
            for k, term
            in obj_vals["terms"].items()
        ]
        # then perform the selected operation to combine the weighted values
        if operation == "sum":
            obj_vals["total"] = sum(weighted_objectives)
        elif operation == "mean":
            obj_vals["total"] = np.mean(weighted_objectives)
        elif operation == "product":
            obj_vals["total"] = np.prod(weighted_objectives)

        # and return the results
        return obj_vals
