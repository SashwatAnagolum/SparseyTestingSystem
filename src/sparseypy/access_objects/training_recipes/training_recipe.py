# -*- coding: utf-8 -*-

"""
Training Recipe: class representing training recipes, which are used to train models.
"""

from datetime import datetime
from typing import Optional
import copy

import torch
from torch.utils.data import DataLoader

from sparseypy.access_objects.preprocessing_stack.preprocessing_stack import PreprocessingStack
from sparseypy.core.data_storage_retrieval import DataStorer
from sparseypy.core.results import TrainingStepResult, TrainingResult

import wandb

class TrainingRecipe:
    """
    TrainingRecipe: class that trains a given model on a 
    particular dataset, using configurations passed in by
    the user.
    Attributes:
        device (torch.device): the device to train the model on.
        model (torch.nn.Module): the model to train.
        optimizer (torch.optim.Optimizer): the optimizer to use.
        train_dataloader (DataLoader): the training dataloader.
        eval_dataloader (DataLoader): the evaluation dataloader.
        preprocessing_stack (PreprocessingStack): the preprocessing stack to apply.
        metrics_list (list[torch.nn.Module]): the metrics to compute.
        metric_config (dict): the configuration for the metrics.
        setup_configs (dict): the setup configurations.
        loss_func (torch.nn.Module): the loss function to use.
        step_resolution (int): the number of batches to train on before logging results.
        batch_index (int): the current batch index.
        training_num_batches (int): the number of batches in the training dataloader.
        training_iterator (iter): the training dataloader iterator.
        eval_num_batches (int): the number of batches in the evaluation dataloader.
        eval_iterator (iter): the evaluation dataloader iterator.
        ds (DataStorer): the data storer object.
        training_results (TrainingResult): the training results object.
        eval_results (TrainingResult): the evaluation results object.
        first_eval (bool): whether this is the first evaluation step.
    """
    def __init__(self, device: torch.device, model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 train_dataloader: DataLoader,
                 eval_dataloader: DataLoader,
                 preprocessing_stack: PreprocessingStack,
                 metrics_list: list[torch.nn.Module],
                 metric_config: dict, setup_configs: dict,
                 loss_func: Optional[torch.nn.Module]) -> None:
        """
        Initializes the TrainingRecipe.
        Args:
            device (torch.device): the device to train the model on.
            model (torch.nn.Module): the model to train.
            optimizer (torch.optim.Optimizer): the optimizer to use.
            train_dataloader (DataLoader): the training dataloader.
            eval_dataloader (DataLoader): the evaluation dataloader.
            preprocessing_stack (PreprocessingStack): the preprocessing stack to apply.
            metrics_list (list[torch.nn.Module]): the metrics to compute.
            metric_config (dict): the configuration for the metrics.
            setup_configs (dict): the setup configurations.
            loss_func (torch.nn.Module): the loss function to use.
            step_resolution (int): the number of batches to train on before logging results.
            """
        self.optimizer = optimizer
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.preprocessing_stack = preprocessing_stack
        self.metrics_list = metrics_list
        self.loss_func = loss_func
        self.setup_configs = setup_configs
        self.device = device

        self.batch_index = 0
        if self.train_dataloader:
            self.training_num_batches = len(self.train_dataloader)
            self.training_iterator = iter(self.train_dataloader)
        else:
            self.training_num_batches = 0
            self.training_iterator = None
        self.eval_num_batches = len(self.eval_dataloader)
        self.eval_iterator = iter(self.eval_dataloader)

        self.ds = DataStorer(metric_config)

        self.training_results = TrainingResult(
            id=wandb.run.id,
            result_type="training",
            metrics=self.metrics_list,
            configs=setup_configs
        )

        self.eval_results = TrainingResult(
            id=wandb.run.id,
            result_type="evaluation",
            metrics=self.metrics_list,
            configs=setup_configs
        )

        self.first_eval = True
        self.ds.create_experiment(self.training_results)


    def step(self, training: bool = True):
        """
        Performs a single step of training or evaluation.

        Args:
            training (bool): whether to perform training (True) or evaluation (False)

        Returns:
            results (TrainingStepResult): the results of this training/evaluation step
            epoch_ended: whether this step has completed the current epoch (in which case
            the full training/evaluation results will be available from get_summary())
        """
        if training:
            num_batches_in_epoch = self.training_num_batches
            data_iterator = self.training_iterator
        else:
            num_batches_in_epoch = self.eval_num_batches
            data_iterator = self.eval_iterator

        if not training and self.first_eval:
            self.first_eval = False
            self.eval_results.start_time = datetime.now()

        data, labels = next(data_iterator)
        labels = labels.to(self.device)

        results = TrainingStepResult(batch_size=data.size(dim=0))

        self.optimizer.zero_grad()

        transformed_data = self.preprocessing_stack(data)
        transformed_data = transformed_data.to(self.device)

        model_output = self.model(transformed_data)

        for metric in self.metrics_list:
            output = metric.compute(
                self.model, transformed_data,
                model_output, training
            )

            # need to add logic for "save only during training/eval" metrics
            results.add_metric(metric.get_name(), output)

        if training:
            if self.loss_func is not None:
                loss = self.loss_func(model_output, labels)
                loss.backward()

            self.optimizer.step()

        self.batch_index += 1

        if self.batch_index == num_batches_in_epoch:
            epoch_ended = True
            self.batch_index = 0

            if training:
                self.training_iterator = iter(self.train_dataloader)
            else:
                self.eval_iterator = iter(self.eval_dataloader)
        else:
            epoch_ended = False

        # at this point the step is finished
        results.mark_finished()

        # log the results for this step and add them to the TrainingResult
        if training:
            self.ds.save_training_step(self.training_results.id, results)
            self.training_results.add_step(results)
        else:
            self.ds.save_evaluation_step(
                self.training_results.id,
                results,
                log_to_wandb=(wandb.run.sweep_id is None)
            )
            self.eval_results.add_step(results)

        return results, epoch_ended


    def get_summary(self, phase: str = "training") -> TrainingResult:
        """
        Returns the completed results for training or evaluation.

        Args:
            phase (str): the phase from which to get results; either
            "training" (default) or "evaluation"

        Returns:
            TrainingResult: the complete results for every step of
            training/evaluation
        """
        if phase == "training":
            self.training_results.mark_finished()
            self.ds.save_training_result(self.training_results)
            self.ds.save_model(
                experiment=wandb.run.id,
                m=copy.deepcopy(self.model).to('cpu'),
                model_config=self.setup_configs["model_config"]
            )
            return self.training_results
        else:
            self.eval_results.mark_finished()
            self.ds.save_evaluation_result(self.eval_results)
            return self.eval_results
