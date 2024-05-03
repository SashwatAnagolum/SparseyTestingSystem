# -*- coding: utf-8 -*-

"""
Test Layer Factory: test cases for the LayerFactory class.
"""

import numpy as np
import pytest
import torch

from sparseypy.core.hpo_objectives.hpo_objective import HPOObjective
from sparseypy.core.results import TrainingResult, TrainingStepResult


class TestHPOObjective:
    """
    TestHPOObjective: a class holding a collection
        of tests focused on the HPOObjective class.
    """
    def test_objective_combination(self) -> None:
        """
        Tests whether the HPOObjective correctly calculates 
        the objective value for some given test data.
        """
        # construct test metric data
        test_metric_data = [
            {
                'basis_set_size': torch.nested.nested_tensor([torch.Tensor([1,2,3,4]), torch.Tensor([5,6,7,8])]),
                'match_accuracy': torch.nested.nested_tensor([torch.Tensor([1,1,1,1]), torch.Tensor([1,1,1,1])]),
                'feature_coverage': torch.nested.nested_tensor([torch.Tensor([1.0]), torch.Tensor([0.1])]),
                'basis_set_size_increase': torch.nested.nested_tensor([torch.Tensor([1,1,1,1]), torch.Tensor([1,1,1,1])])
            },
            {
                'basis_set_size': torch.nested.nested_tensor([torch.Tensor([1,1,1,1]), torch.Tensor([1,1,1,1])]),
                'match_accuracy': torch.nested.nested_tensor([torch.Tensor([1,1,1,1]), torch.Tensor([1,1,1,1])]),
                'feature_coverage': torch.nested.nested_tensor([torch.Tensor([1.0]), torch.Tensor([1.0])]),
                'basis_set_size_increase': torch.nested.nested_tensor([torch.Tensor([1,1,1,1]), torch.Tensor([1,1,1,1])])
            },
            # Add more entries as needed
        ]

        test_result = TrainingResult("test", result_type="evaluation", metrics=[])
        for test_metric in test_metric_data:
            tsr = TrainingStepResult()
            tsr.metrics = test_metric
            test_result.results.append(tsr)
        # construct test HPO configuration
        hpo_config = {
            "optimization_objective": {
                "combination_method": "mean",
                "objective_terms": [
                    {'metric': {"name": "basis_set_size"}, "weight": 1.0},
                    {'metric': {"name": "match_accuracy"}, "weight": 1},
                    {'metric': {"name": "feature_coverage"}, "weight": 1.0},
                    {'metric': {"name": "basis_set_size_increase"}, "weight": 1}
                ]
            }
        }

        # Initialize HPOObjective with the configuration
        hpo_objective = HPOObjective(hpo_config)

        # Calculate combined metric
        combined_metric = hpo_objective.combine_metrics(test_result)
        # assert metric values
        np.testing.assert_almost_equal(combined_metric['total'], 1.38125)
        test_metric_data = [
            {
                'match_accuracy': torch.nested.nested_tensor([torch.Tensor([5.0])]),
                'basis_set_size': torch.nested.nested_tensor([torch.Tensor([3,5,5,7]), torch.Tensor([1,9,1,9])])
            },
            {
                'match_accuracy': torch.nested.nested_tensor([torch.Tensor([3.0])]),
                'basis_set_size': torch.nested.nested_tensor([torch.Tensor([1,5,1,5]), torch.Tensor([2,4,2,4])])
            },
            # Add more entries as needed
        ]

        # construct test HPO configuration
        hpo_config = {
            "optimization_objective": {
                "combination_method": "mean",
                "objective_terms": [
                    {'metric': {"name": "match_accuracy"}, "weight": 1.0},
                    {'metric': {"name": "basis_set_size"}, "weight": 0.5}
                ]
            }
        }

        # Initialize HPOObjective with the configuration
        hpo_objective = HPOObjective(hpo_config)

        test_result = TrainingResult("test", result_type="evaluation", metrics=[])
        for test_metric in test_metric_data:
            tsr = TrainingStepResult()
            tsr.metrics = test_metric
            test_result.results.append(tsr)

        # Calculate combined metric
        combined_metric = hpo_objective.combine_metrics(test_result)
        np.testing.assert_almost_equal(combined_metric['total'], 3)
