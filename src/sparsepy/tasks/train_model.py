# -*- coding: utf-8 -*-

"""
Train Model: script to train models.
"""


import torch


from sparsepy.access_objects.models.model_builder import ModelBuilder
from sparsepy.access_objects.training_recipes.training_recipe_builder import TrainingRecipeBuilder


def train_model(model_config: dict, trainer_config: dict,
                preprocessing_config: dict, dataset_config: dict):
    """
    Builds a model using the model_config, and trains
    it using the trainer built using trainer_config on 
    the dataset built using dataset_config, with preprocessing
    defined in preprocessing_config.
    """
    model = ModelBuilder.build_model(model_config)

    trainer = TrainingRecipeBuilder.build_training_recipe(
        model, dataset_config, preprocessing_config,
        trainer_config
    )

    for epoch in range(trainer_config['training']['num_epochs']):
        is_epoch_done = False

        while not is_epoch_done:
            output, is_epoch_done = trainer.step()

        # for layer in range(len(model.layers)):
        #     for mac in range(len(model.layers[layer].mac_list)):
        #         max_codes = model.layers[layer].mac_list[mac].weights.shape[2] ** model.layers[layer].mac_list[mac].weights.shape[0]

        #         print(f'Layer {layer} | MAC {mac} | Stored codes: {len(output[layer][mac])} | Max codes possible: {max_codes}')
