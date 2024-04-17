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
from sparseypy.access_objects.datasets import PreprocessedDataset, InMemoryDataset
from sparseypy.access_objects.models.model_builder import ModelBuilder


class TrainingRecipeBuilder:
    """
    TrainingRecipeBuilder: builder class for TrainingRecipe
    objects.
    """
    @staticmethod
    def build_training_recipe(model_config: dict, 
        dataset_config: dict, preprocessing_config: dict,
        train_config: dict) -> TrainingRecipe:
        """
        Builds the training recipe object using the
        passed in config information.

        Args:
            model_config (dict): the config info
                for the model to be trained.
            dataset_config (dict): the config info
                for the dataset to be used.
            preprocessing_config (dict): the config info
                for the preprocessing stack to be applied
                onto the data.
            train_config (dict): the config info for
                the training recipe to use to train the
                model.

        Returns:
            (TrainingRecipe): the constructed
                TrainingRecipe object.
        """
        device = torch.device('cuda' if train_config['use_gpu'] else 'cpu')

        model = ModelBuilder.build_model(model_config)
        model.to(device)

        preprocessing_stack = PreprocessingStack(preprocessing_config)

        optimizer = OptimizerFactory.create_optimizer(
            train_config['optimizer']['name'],
            **train_config['optimizer']['params'],
            model=model
        )

        dataset = DatasetFactory.create_dataset(
            dataset_config['dataset_type'],
            **dataset_config['params']
        )

        # if a preprocessed dataset then wrap the dataset
        if dataset_config['preprocessed'] is True:
            preprocessed_dataset_stack = PreprocessingStack(
                dataset_config['preprocessed_stack']
            )

            dataset = PreprocessedDataset(
                dataset, preprocessed_dataset_stack,
                dataset_config['preprocessed_temp_dir'],
                dataset_config['save_to_disk']
            )
        
        if dataset_config['in_memory']:
            dataset = InMemoryDataset(
                dataset, dataset_config['load_lazily']
            )

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
                metric_name=train_config['loss']['name'],
                **train_config['loss']['params']
            )
        else:
            loss_func = None

        #loss_func = None
             
        # store the configs inside the finished TrainingRecipe for later saving
        setup_configs = {
            'dataset_config': dataset_config,
            'model_config': model_config,
            'preprocessing_config': preprocessing_config,
            'training_recipe_config': train_config
        }

        return TrainingRecipe(
            device, model, optimizer, dataloader,
            preprocessing_stack, metrics_list,
            train_config['metrics'], setup_configs,
            loss_func,
            train_config['training']['step_resolution']
        )
