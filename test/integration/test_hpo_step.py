# -*- coding: utf-8 -*-

"""
Test HPO Strategy: tests covering the HPOStrategy class, particularly 
the get_next_parameters method to ensure it returns parameters present in 
the last experiment result.
"""

import pytest
from sparsepy.core.hpo_stratagies.hpo_stratagy import HPOStrategy
from sparsepy.core.hpo_stratagies.experiment_result import ExperimentResult

class TestHPOStrategy:
    """
    TestHPOStrategy: A class for testing the HPOStrategy implementations.
    Specifically, it tests the get_next_parameters method.
    """

    @pytest.fixture
    def setup_experiment_result(self) -> ExperimentResult:
        """
        Creates a mock ExperimentResult instance for testing.

        Returns:
            ExperimentResult: A mock instance of ExperimentResult with predefined values.
        """
        # Mock experiment result
        experiment_result = ExperimentResult({
            'param1': 0.5,
            'param2': 3,
            'param3': 'value'
            # Add more parameters as needed
        })

        return experiment_result

    @pytest.fixture
    def hpo_strategy(self) -> HPOStrategy:
        """
        Creates an instance of HPOStrategy for testing.

        Returns:
            HPOStrategy: An instance of HPOStrategy.
        """
        return HPOStrategy()

    def test_get_next_parameters(self, hpo_strategy: HPOStrategy, setup_experiment_result: ExperimentResult) -> None:
        """
        Tests that the get_next_parameters method of HPOStrategy 
        returns a dictionary of parameters, and each of these parameters 
        is present in the last experiment result.

        Args:
            hpo_strategy: An instance of HPOStrategy for testing.
            setup_experiment_result: A mock instance of ExperimentResult with predefined values.
        """
        next_params = hpo_strategy.get_next_parameters(setup_experiment_result)
        assert isinstance(next_params, dict), "Returned value is not a dictionary"

        for param in next_params:
            assert param in setup_experiment_result.parameters, f"Parameter '{param}' not found in experiment result"
