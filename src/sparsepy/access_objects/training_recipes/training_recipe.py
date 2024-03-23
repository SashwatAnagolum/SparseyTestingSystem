# -*- coding: utf-8 -*-

"""
Training Recipe: class representing training recipes, which are used to train models.
"""

from datetime import datetime
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Transform

from sparsepy.access_objects.preprocessing_stack.preprocessing_stack import PreprocessingStack
from sparsepy.core.data_storage_retrieval import DataStorer
from sparsepy.core.results import TrainingStepResult, TrainingResult

import wandb

class TrainingRecipe:
    def __init__(self, model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 dataloader: DataLoader,
                 preprocessing_stack: PreprocessingStack,
                 metrics_list: list[torch.nn.Module],
                 metric_config: dict,
                 database_resolution: str,
                 loss_func: Optional[torch.nn.Module],
                 step_resolution: Optional[int] = None) -> None:
        self.optimizer = optimizer
        self.model = model
        self.dataloader = dataloader
        self.preprocessing_stack = preprocessing_stack
        self.metrics_list = metrics_list
        self.loss_func = loss_func

        if step_resolution is None:
            self.step_resolution = 1 #len(self.dataloader)
        else:
            self.step_resolution = step_resolution

        self.batch_index = 0
        self.num_batches = len(self.dataloader)
        self.iterator = iter(self.dataloader)

        self.ds = DataStorer(metric_config)

        # BUG need to have logged in to W&B by the time this is executed
        # BUG reporting fake value currently
        self.training_results = TrainingResult(wandb.run.id, "training", self.step_resolution, self.metrics_list)
        self.eval_results = TrainingResult(wandb.run.id, "evaluation", self.step_resolution, self.metrics_list)
        self.first_eval = True

        self.ds.create_experiment(self.training_results)


    def step(self, training: bool = True):
        if self.batch_index + self.step_resolution >= self.num_batches:
            num_batches_in_step = self.num_batches - self.batch_index
        else:
            num_batches_in_step = self.step_resolution

        if not training and self.first_eval:
                self.first_eval = False
                self.eval_results.start_time = datetime.now()

        #results = []
        #if training:
        #    results = TrainingStepResult(self.step_resolution)
        #else:
        #    # need to be able to access dataset name from TR
        #    # BUG incorrect dataset name saved
        #    results = ProcessStepResult(self.step_resolution)
        results = TrainingStepResult(self.step_resolution)

        for _ in range(num_batches_in_step):
            data, labels = next(self.iterator)



            self.optimizer.zero_grad()

            transformed_data = self.preprocessing_stack(data)

            transformed_data = transformed_data.view(
                (transformed_data.shape[0], -1, 1, 1)
            )

            model_output = self.model(transformed_data)

            #result = {}

            for metric in self.metrics_list:
                output = metric.compute(
                    self.model, transformed_data,
                    model_output, training
                )

                #result[metric.__class__.__name__] = output
                # need to add logic for "save only during training/eval" metrics
                results.add_metric(metric.get_name(), output)

            if training:
                if self.loss_func is not None:
                    loss = self.loss_func(model_output, labels)
                    loss.backward()

                self.optimizer.step()

            #print("\n" + "\n" + "\n")
            #results.append(result)

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
            
        # at this point the step is finished
        results.mark_finished()

        # log the results for this step and add them to the TrainingResult
        if training:
            self.ds.save_training_step(self.training_results.id, results)
            self.training_results.add_step(results)
        else:
            #self.ds.save_evaluation_step(self.evaluation_results.id, results)
            self.ds.save_evaluation_step(self.training_results.id, results)
            self.eval_results.add_step(results)

        return results, epoch_ended

    def get_summary(self, phase: str = "training") -> TrainingResult:
        if phase == "training":
            self.training_results.mark_finished()
            self.ds.save_training_result(self.training_results)
            return self.training_results
        else:
            self.eval_results.mark_finished()
            self.ds.save_evaluation_result(self.eval_results)
            return self.eval_results
