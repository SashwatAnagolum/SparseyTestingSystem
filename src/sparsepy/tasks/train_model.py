# -*- coding: utf-8 -*-

"""
Train Model: script to train models.
"""


import torch


from sparsepy.access_objects.models.model_builder import ModelBuilder
from sparsepy.access_objects.trainers.trainer_builder import TrainerBuilder


def train_model(model_config, trainer_config,
                dataset_config, preprocessing_config):
    """
    Builds a model using the model_config, and trains
    it using the trainer built using trainer_config on 
    the dataset built using dataset_config, with preprocessing
    defined in preprocessing_config.
    """
    model = ModelBuilder.build_model(model_config)

    trainer = TrainerBuilder.build_trainer(
        model, dataset_config, preprocessing_config,
        trainer_config
    )

    print(model)
    print(trainer)
