# -*- coding: utf-8 -*-

"""
Trainer: class representing trainers, which are used to train models.
"""


from typing import Optional

import torch
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Transform

from sparsepy.access_objects.preprocessing_stack.preprocessing_stack import PreprocessingStack


class Trainer:
    def __init__(self, model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 dataloader: DataLoader,
                 preprocessing_stack: list[Transform],
                 metrics_list: list[torch.nn.Module],
                 loss_func: Optional[torch.nn.Module]) -> None:
        self.optimizer = optimizer
        self.model = model
        self.dataloader = dataloader
        self.preprocessing_stack = preprocessing_stack
        self.metrics_list = metrics_list

        if loss_func is None:
            self.loss_func = torch.nn.Identity()
        else:
            self.loss_func = loss_func


    def step(self):
        data, labels = next(iter(self.dataloader))

        model_output = self.model(data)

        if self.loss_func is not None:
            optimizer_closure = lambda: self.loss_func(
                model_output, labels
            )
        else:
            optimizer_closure = lambda: model_output

        output = self.optimizer.step(optimizer_closure)

        print(output)
