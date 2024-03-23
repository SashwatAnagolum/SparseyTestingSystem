# -*- coding: utf-8 -*-

"""
Build Trainer: class to build trainers for models.
"""


import torch
from torch.utils.data import DataLoader

from sparseypy.access_objects.training_recipes.training_recipe import TrainingRecipe
from sparseypy.core.optimizers.optimizer_factory import OptimizerFactory
from sparseypy.access_objects.datasets.dataset_factory import DatasetFactory
from sparseypy.access_objects.preprocessing_stack.preprocessing_stack import PreprocessingStack
from sparseypy.core.metrics.metric_factory import MetricFactory
from sparseypy.access_objects.datasets.preprocessed_dataset import PreprocessedDataset

class TrainingRecipeBuilder:
    @staticmethod
    def build_training_recipe(model: torch.nn.Module, 
                      dataset_config: dict,
                      preprocessing_config: dict,
                      train_config: dict) -> TrainingRecipe:
        optimizer = OptimizerFactory.create_optimizer(
            train_config['optimizer']['name'],
            **train_config['optimizer']['params'],
            model=model
        )

        preprocessing_stack = PreprocessingStack(preprocessing_config)

        dataset = DatasetFactory.create_dataset(
            dataset_config['dataset_type'],
            **dataset_config['params']
        )
        
        # if a preprocessed dataset then wrap the dataset and cancel the other preprocessing stack
        if dataset_config['preprocessed'] is True:
            preprocessed_dataset_stack = PreprocessingStack(dataset_config['preprocessed_stack'])
            dataset = PreprocessedDataset(dataset, preprocessed_dataset_stack, dataset_config['preprocessed_temp_dir'])
            #print("PREPROCESSED")

        dataloader = DataLoader(
            dataset=dataset, **train_config['dataloader']
        )

        metrics_list = []

        # BUG this is a bad way to do metric initialization and will break every time we add a Metric parameter
        # we should revisit it
        for metric_config in train_config['metrics']:
            metric = MetricFactory.create_metric(
                metric_config['name'],
                #**metric_config['params'],
                model=model,
                reduction=metric_config['reduction'], # WARNING this formulation assumes all Metrics support a reduction constructor parameter
                best_value=metric_config['best_value']
            )

            metrics_list.append(metric)

        if 'loss' in train_config:
            loss_func = MetricFactory.create_metric(
                train_config['loss']['name'],
                train_config['loss']['params']
            )
        else:
             loss_func = None

        #loss_func = None

        return TrainingRecipe(
            model, optimizer, dataloader,
            preprocessing_stack, metrics_list,
            train_config['metrics'], # ensure this will also save metrics added only as HPO objectives!
            loss_func,
            train_config['training']['step_resolution']
        )
