# -*- coding: utf-8 -*-

"""
Test Sparsey Dataset Configs: tests covering the config files for 
    customizing the structure of Sparsey datasets.
"""


import os

import pytest

from sparseypy.cli.config_validation.validate_config import get_config_info
from sparseypy.access_objects.models.model_builder import ModelBuilder
from sparseypy.access_objects.training_recipes.training_recipe_builder import (
    TrainingRecipeBuilder, TrainingRecipe
)


class TestTrainer:
    """
    TestTrainingRecipe: class containing a collection
    of tests related to training recipes.
    """
    @pytest.fixture
    def valid_trainer(self) -> TrainingRecipe:
        """
        Returns a valid training recipe object.

        Returns:
            (TrainingRecipe): a valid training recipe
        """
        dataset_config = get_config_info(
            './test/integration/resources/dataset.yaml'
        )

        preprocessing_config = get_config_info(
            './test/integration/resources/preprocessing.yaml'
        )

        trainer_config = get_config_info(
            './test/integration/resources/trainer.yaml'
        )

        model_config = get_config_info(
            './test/integration/resources/network.yaml'
        )

        model = ModelBuilder.build_model(model_config)

        trainer = TrainingRecipeBuilder.build_training_recipe(
            model, dataset_config, preprocessing_config,
            trainer_config
        )

        return trainer
    

    def test_trainer_step_output(self, trainer: TrainingRecipe):
        """
        Tests if the trainer returns the expected dictionary
        of metric results after every step.
        """
        results, epoch_done = trainer.step(training=True)

        assert (
            (len(results) == 1) and
            (len(results[0]) == 2) and 
            (isinstance(results[0], dict)) and
            ('BasisAverageMetric' in results[0]) and
            ('NumActivationsMetric' in results[0])
        )