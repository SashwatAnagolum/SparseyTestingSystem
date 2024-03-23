# -*- coding: utf-8 -*-

"""
Test HPO Strategy: tests covering the HPOStrategy class, particularly 
the get_next_parameters method to ensure it returns parameters present in 
the last experiment result.
"""

import pytest
from sparseypy.core.hpo_stratagies.hpo_stratagy import HPOStrategy
from sparseypy.core.hpo_stratagies.experiment_result import ExperimentResult
from sparseypy.core.hpo_strategies.exceptions import UntrackableParameterException

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

    def test_untrackable_parameter(self, hpo_strategy: HPOStrategy, setup_experiment_result: ExperimentResult) -> None:
        """
        Tests that the get_next_parameters method raises an exception when asked to track
        a parameter not present in the last experiment result.

        Args:
            hpo_strategy: An instance of HPOStrategy for testing.
            setup_experiment_result: A mock instance of ExperimentResult with predefined values.
        """
        # Here we manually add an untrackable parameter to the HPOStrategy instance
        # which simulates the scenario where HPOStrategy is asked to get a next parameter
        # that wasn't part of the last experiment results.
        hpo_strategy.untrackable_param = 'untrackable_param_not_in_result'
        
        with pytest.raises(UntrackableParameterException) as exc_info:
            hpo_strategy.get_next_parameters(setup_experiment_result)
        
        assert "untrackable parameter" in str(exc_info.value), "UntrackableParameterException not raised for untrackable parameter"
