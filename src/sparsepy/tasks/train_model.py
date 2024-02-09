# -*- coding: utf-8 -*-

"""
Train Model: script to train models.
"""


import pprint

import torch

from sparsepy.access_objects.models.model_builder import ModelBuilder
from sparsepy.access_objects.training_recipes.training_recipe_builder import (
    TrainingRecipeBuilder
) 


def train_model(model_config: dict, trainer_config: dict,
                preprocessing_config: dict, dataset_config: dict):
    """
    Builds a model using the model_config, and trains
    it using the trainer built using trainer_config on 
    the dataset built using dataset_config, with preprocessing
    defined in preprocessing_config.

    Args:
        model_config (dict): config info to build the model.
        trainer_config (dict): config info to build the trainer.
        preprocessing_config (dict): config info to build the
            preprocessing stack.
        dataset_config (dict): config info to build the dataset
            to train on.
    """
    model = ModelBuilder.build_model(model_config)

    trainer = TrainingRecipeBuilder.build_training_recipe(
        model, dataset_config, preprocessing_config,
        trainer_config
    )

    for epoch in range(trainer_config['training']['num_epochs']):
        is_epoch_done = False
        model.train()

        while not is_epoch_done:
            output, is_epoch_done = trainer.step(training=True)
            print("\n\nTraining results\n--------------------")
            pprint.pprint(output)

        model.eval()
        is_epoch_done = False

        while not is_epoch_done:
            output, is_epoch_done = trainer.step(training=False)
            print("\n\nEvaluation results\n--------------------")
            pprint.pprint(output)
