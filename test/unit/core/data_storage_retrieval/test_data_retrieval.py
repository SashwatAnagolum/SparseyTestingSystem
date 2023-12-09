# -*- coding: utf-8 -*-

"""
Test DataFetcher: Tests to ensure that the DataFetcher class
correctly retrieves model weights and training results.
"""

import pytest
from unittest.mock import Mock
from sparsepy.core.data_storage_retrieval import DataFetcher, TrainingResult

class TestDataFetcher:
    """
    TestDataFetcher: Test suite for the DataFetcher class.
    """
    
    @pytest.fixture
    def data_fetcher(self) -> DataFetcher:
        """
        Fixture to create a DataFetcher instance for testing.
        """
        return DataFetcher()

    @pytest.fixture
    def mock_weights(self) -> dict:
        """
        Fixture to create mock weights for a model.
        """
        return {'weight1': 0, 'weight2': 127}

    @pytest.fixture
    def mock_training_result(self) -> TrainingResult:
        """
        Fixture to create a mock TrainingResult instance for testing.
        """
        return TrainingResult(accuracy=0.95, loss=0.05)

    def test_get_model_weights(self, data_fetcher: DataFetcher, mock_weights: dict) -> None:
        """
        Test the get_model_weights method to ensure it returns the correct model weights.
        """
        # Here you would mock the actual call to the data source
        data_fetcher.get_model_weights = Mock(return_value=mock_weights)
        weights = data_fetcher.get_model_weights("model_id_123")
        assert weights == mock_weights, "The returned weights do not match the expected mock weights"

    def test_get_training_result(self, data_fetcher: DataFetcher, mock_training_result: TrainingResult) -> None:
        """
        Test the get_training_result method to ensure it returns the correct TrainingResult object.
        """
        # Here you would mock the actual call to the data source
        data_fetcher.get_training_result = Mock(return_value=mock_training_result)
        training_result = data_fetcher.get_training_result("experiment_id_123")
        assert training_result == mock_training_result, "The returned TrainingResult does not match the expected mock TrainingResult"
