# -*- coding: utf-8 -*-

"""
Training Recipe: class representing training recipes, which are used to train models.
"""


from typing import Optional

import torch
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Transform

from sparsepy.access_objects.preprocessing_stack.preprocessing_stack import PreprocessingStack


class TrainingRecipe:
    def __init__(self, model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 dataloader: DataLoader,
                 preprocessing_stack: PreprocessingStack,
                 metrics_list: list[torch.nn.Module],
                 loss_func: Optional[torch.nn.Module],
                 step_resolution: Optional[int] = None) -> None:
        self.optimizer = optimizer
        self.model = model
        self.dataloader = dataloader
        self.preprocessing_stack = preprocessing_stack
        self.metrics_list = metrics_list
        self.loss_func = loss_func

        if step_resolution is None:
            self.step_resolution = len(self.dataloader)
        else:
            self.step_resolution = step_resolution

        self.batch_index = 0
        self.num_batches = len(self.dataloader)
        self.iterator = iter(self.dataloader)


    def step(self, training: bool = True):
        if self.batch_index + self.step_resolution >= self.num_batches:
            num_batches_in_step = self.num_batches - self.batch_index
        else:
            num_batches_in_step = self.step_resolution

        results = []

        for _ in range(num_batches_in_step):
            data, labels = next(self.iterator)

            self.optimizer.zero_grad()

            transformed_data = self.preprocessing_stack(data)

            transformed_data = transformed_data.view(
                (transformed_data.shape[0], -1, 1, 1)
            )

            model_output = self.model(transformed_data)

            result = {}

            for metric in self.metrics_list:
                output = metric.compute(
                    self.model, transformed_data,
                    model_output, training
                )

                result[metric.__class__.__name__] = output

            if training:
                if self.loss_func is not None:
                    loss = self.loss_func(model_output, labels)
                    loss.backward()

                self.optimizer.step()

            results.append(result)
            

        self.batch_index += num_batches_in_step

        if self.batch_index == self.num_batches:
            epoch_ended = True
            self.batch_index = 0
            self.iterator = iter(self.dataloader)
        else:
            epoch_ended = False

        # stored_codes = [
        #    [mac.stored_codes for mac in layer.mac_list]
        #    for layer in self.model.layers
        # ]

        return results, epoch_ended