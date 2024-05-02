# -*- coding: utf-8 -*-

"""
Test DataStorer: Tests for ensuring that the DataStorer class
correctly interacts with model saving, training result saving, 
and artifact creation in Weights & Biases.
"""

import pytest
from unittest.mock import Mock, patch
from sparseypy.core.data_storage_retrieval import DataStorer
from sparseypy.access_objects.models.model import Model
from sparseypy.core.results.training_result import TrainingResult
import wandb

@pytest.fixture
def mock_wandb():
    with patch('your_module.wandb') as mock:
        yield mock

class TestDataStorer:
    """
    TestDataStorer: Test suite for the DataStorer class.
    """

    @pytest.fixture
    def model(self) -> Model:
        """
        Fixture to create a mock Model instance for testing.
        """
        model = Mock(spec=Model)
        model.weights = {'weight1': 1, 'weight2': 127}
        return model

    @pytest.fixture
    def training_result(self) -> TrainingResult:
        """
        Fixture to create a mock TrainingResult instance for testing.
        """
        result = Mock(spec=TrainingResult)
        result.data = {'accuracy': 0.95, 'loss': 0.05}
        return result

    @pytest.fixture
    def content_dict(self) -> dict:
        """
        Fixture to create a mock content dictionary for artifact creation.
        """
        return {'key': 'value'}

    @pytest.fixture
    def data_storer(self, mock_wandb) -> DataStorer:
        """
        Fixture to create a DataStorer instance with mocked wandb dependency.
        """
        return DataStorer()

    def test_save_model(self, data_storer: DataStorer, model: Model, mock_wandb) -> None:
        """
        Test the save_model method to ensure it handles the Model object correctly and logs it to wandb.
        """
        data_storer.save_model(model)
        mock_wandb.log.assert_called_once_with({'model': model})

    def test_save_training_result(self, data_storer: DataStorer, training_result: TrainingResult, mock_wandb) -> None:
        """
        Test the save_training_result method to ensure it handles the TrainingResult object correctly and logs it to wandb.
        """
        data_storer.save_training_result(training_result)
        mock_wandb.log.assert_called_once_with(training_result.data)

    def test_create_artifact(self, data_storer: DataStorer, content_dict: dict, mock_wandb) -> None:
        """
        Test the create_artifact method to ensure it handles the content dictionary correctly and creates a wandb.Artifact.
        """
        artifact = data_storer.create_artifact(content_dict)
        mock_wandb.Artifact.assert_called_once_with()
        mock_wandb.log_artifact.assert_called_once_with(artifact)
