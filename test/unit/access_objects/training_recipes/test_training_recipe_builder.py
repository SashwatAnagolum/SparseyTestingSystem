import pytest
import torch
import os
import wandb
from sparseypy.access_objects.training_recipes.training_recipe_builder import TrainingRecipeBuilder
from sparseypy.core.data_storage_retrieval.data_storer import DataStorer
from sparseypy.core.optimizers.optimizer_factory import OptimizerFactory
from sparseypy.core.metrics.metric_factory import MetricFactory
from sparseypy.core.metrics.metrics import Metric

class TestTrainingRecipeBuilder:
    """
    Tests for the TrainingRecipeBuilder class.
    """
    def setup_method(self):
        """ Setup common configuration for each test """
        self.model_config = {
            'layers': [
                {
                    'name': 'sparsey',
                    'params': {
                        'autosize_grid': True, 
                        'grid_layout': 'hex',
                        'num_macs': 25, 
                        'num_cms_per_mac': 5, 
                        'num_neurons_per_cm': 5,
                        'mac_grid_num_rows': 5, 
                        'mac_grid_num_cols': 5,
                        'mac_receptive_field_size': 1.5, 
                        'prev_layer_num_cms_per_mac': 12,
                        'prev_layer_num_neurons_per_cm': 10,
                        'prev_layer_mac_grid_num_rows': 4,
                        'prev_layer_mac_grid_num_cols': 6,
                        'prev_layer_num_macs': 24, 
                        'prev_layer_grid_layout': 'rect',
                        'layer_index': 6, 
                        'sigmoid_phi': 0.1, 
                        'sigmoid_lambda': 0.9,
                        'saturation_threshold': 0.8,
                        'permanence_steps': 0.1, 
                        'permanence_convexity': 0.1,
                        'activation_threshold_min': 0.4, 
                        'activation_threshold_max': 0.8,
                        'min_familiarity': 0.5, 
                        'sigmoid_chi': 1.2
                    }
                }
            ]
        }

        self.dataset_config = {
            'dataset_type': 'image',
            'description': 'MNIST dataset for testing purposes',
            'params': {
                'data_dir': './demo/sample_mnist_dataset',
                'image_format': '.png'
            },
            'preprocessed': False,
            'preprocessed_stack': {
                'transform_list': [
                    {'name': 'resize', 'params': {'size': [8, 8], 'antialias': True}},
                    {'name': 'binarize', 'params': {'binarize_threshold': 0.5}},
                    {'name': 'skeletonize', 'params': {'sigma': 3}}
                ]
            },
            'in_memory': True,
            'load_lazily': False,
            'save_to_disk': True
        }

        self.preprocessing_config = {
            'transform_list': [
                {'name': 'resize', 'params': {'size': [8, 8], 'antialias': True}},
                {'name': 'to_dtype', 'params': {'dtype': 'float32', 'scale': True}}
            ]
        }

        self.train_config = {
            'optimizer': {
                'name': 'hebbian',
                'params': {}
            },
            'metrics': [
                {'name': 'basis_set_size', 'save': False, 'best_value': 'max_by_layerwise_mean', 'reduction': 'mean', 'params': {}},
                {'name': 'feature_coverage', 'save': False, 'best_value': 'max_by_layerwise_mean', 'reduction': 'mean', 'params': {}},
            ],
            'training': {
                'num_epochs': 1,
                'dataloader': {
                    'batch_size': 4,
                    'shuffle': True
                },
            },
            'eval': {
                'dataloader': {
                    'batch_size': 4,
                    'shuffle': False
                }
            },
            'use_gpu': False
        }

        self.system_config = {
            "wandb": {
                "api_key": os.getenv('WANDB_API_KEY'),
                "project_name": "sparsey_testing_project",
                "save_models": True,
                "save_locally": True,
                "data_resolution": 2,
                "silent": True
            },
            "database": {
                "read_database": "firestore",
                "write_databases": [
                    {
                        "name": "firestore",
                        "table_names": {
                            "hpo_runs": "hpo_runs",
                            "experiments": "experiments",
                            "batches": "batches",
                            "models": "models",
                            "model_registry": "model_registry"
                        },
                        "firebase_service_key_path": os.getenv('FIREBASE_CONFIG_FILE'),
                        "data_resolution": 2,
                        "save_models": False,
                        "batch_size": 64
                    }
                ]
            },
            "print_error_stacktrace": False
        }

    def test_correct_optimizer_construction(self):
        """
        Tests the TrainingRecipeBuilder's capability to correctly construct
        the optimizer as specified in the user's configuration file.

        Test Case ID: TC-10-01
        """
        DataStorer.configure(self.system_config)

        wandb.init(
            project=self.system_config["wandb"]["project_name"],
            allow_val_change=True,
            job_type="train",
            config={
                'dataset': self.dataset_config,
                'model': self.model_config,
                'training_recipe': self.train_config,
                'preprocessing': self.preprocessing_config
            }
        )

        training_recipe = TrainingRecipeBuilder.build_training_recipe(
            self.model_config, self.dataset_config, self.dataset_config,
            self.preprocessing_config, self.train_config
        )
        assert isinstance(training_recipe.optimizer, torch.optim.Optimizer), \
            "Optimizer was not constructed correctly"
