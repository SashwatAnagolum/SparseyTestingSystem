# -*- coding: utf-8 -*-

"""
Train Model: script to train models.
"""


import pprint

import torch
import wandb

from sparsepy.tasks.api_login import log_in
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
    log_in()

    wandb.init(
        project="wandb_run_log_testing", entity="sparsey-testing-system" # FIXME add W&B project name to trainer config
    )

    model = ModelBuilder.build_model(model_config)

    trainer = TrainingRecipeBuilder.build_training_recipe(
        model, dataset_config, preprocessing_config,
        trainer_config
    )

    for epoch in range(trainer_config['training']['num_epochs']):
        is_epoch_done = False
        model.train()
        batch_number = 1

        while not is_epoch_done:
            output, is_epoch_done = trainer.step(training=True)
            print(f"\n\nTraining results - INPUT {batch_number}\n--------------------")
            pprint.pprint(output.get_metrics())
            batch_number+=1

        model.eval()
        is_epoch_done = False
        batch_number = 1

        while not is_epoch_done:
            # validate this logic VS the design of our EvaluationResult
            # this looks like old-style logic for which we should remove the "while"
            output, is_epoch_done = trainer.step(training=False)
            print(f"\n\nEvaluation results - INPUT {batch_number}\n--------------------")
            pprint.pprint(output.get_metrics())
            batch_number+=1

        # print summary here in model script
        # if not printing you still need to call this to finalize the results
        # FIXME confirm this is the correct location
        # FIXME we are not correctly updating eval results in the DB if we do this
        # 
        trainer.get_summary()
        wandb.finish()
