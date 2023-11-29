# -*- coding: utf-8 -*-

"""
Build Trainer: class to build trainers for models.
"""


import torch

from sparsepy.access_objects.trainers.trainer import Trainer
from sparsepy.core.optimizers.optimizer_factory import OptimizerFactory


class TrainerBuilder:
    @staticmethod
    def build_trainer(model: torch.nn.Module, 
                      dataset_config: dict,
                      preprocessing_config: dict,
                      trainer_config: dict) -> Trainer:
        optimizer = OptimizerFactory.create_optimizer(
            trainer_config['optimizer']['name'],
            **trainer_config['optimizer']['params'],
            params=model.parameters()
        )

        preprocessing_stack = None
        metrics_list = []
        loss_func = None

        return Trainer(
            model, optimizer, dataloader,
            preprocessing_stack, metrics_list,
            loss_func
        )


