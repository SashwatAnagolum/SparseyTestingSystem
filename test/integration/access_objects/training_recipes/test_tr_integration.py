import os
import pytest
import torch
import wandb
from sparseypy.access_objects.training_recipes.training_recipe_builder import TrainingRecipeBuilder
from sparseypy.core.data_storage_retrieval.data_storer import DataStorer
from sparseypy.core.metrics.feature_coverage import FeatureCoverageMetric
from sparseypy.core.metrics.match_accuracy import MatchAccuracyMetric

class TestTRIntegration:
    """
    Integration tests for the TrainingRecipe class.
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
                        'num_macs': 9, 
                        'num_cms_per_mac': 2, 
                        'num_neurons_per_cm': 2,
                        'mac_grid_num_rows': 3, 
                        'mac_grid_num_cols': 3,
                        'mac_receptive_field_size': 0.6, 
                        'prev_layer_num_cms_per_mac': 4,
                        'prev_layer_num_neurons_per_cm': 4,
                        'prev_layer_mac_grid_num_rows': 1,
                        'prev_layer_mac_grid_num_cols': 1,
                        'prev_layer_num_macs': 1, 
                        'prev_layer_grid_layout': 'rect',
                        'layer_index': 0, 
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
                    {'name': 'resize', 'params': {'size': [8,8], 'antialias': True}},
                    {'name': 'binarize', 'params': {'binarize_threshold': 0.5}},
                    {'name': 'skeletonize', 'params': {'sigma': 3}},
                    {'name': 'sparsey_input_reshape', 'params': {}}
                ]
            },
            'in_memory': True,
            'load_lazily': False,
            'save_to_disk': True
        }

        self.preprocessing_config = {
            'transform_list': [
                {'name': 'resize', 'params': {'size': [4, 4], 'antialias': True}},
                {'name': 'to_dtype', 'params': {'dtype': 'float32', 'scale': True}},
                {'name': 'sparsey_input_reshape', 'params': {}},
                {'name': 'resize', 'params': {'size': [1, 16], 'antialias': True}},
            ]
        }

        self.train_config = {
            'optimizer': {
                'name': 'hebbian',
                'params': {}
            },
            'metrics': [
                {'name': 'match_accuracy', 'save': False, 'best_value': 'max_by_layerwise_mean', 'reduction': 'mean', 'params': {}},
                {'name': 'feature_coverage', 'save': False, 'best_value': 'max_by_layerwise_mean', 'reduction': 'mean', 'params': {}},
            ],
            'eval': {
                'dataloader': {
                    'batch_size': 16,
                    'shuffle': True
                },
            },
            'training': {
                'dataloader': {
                    'batch_size': 1,
                    'shuffle': True
                },
                'num_epochs': 1,
                #'step_resolution': 10
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

    def test_tr_step(self):
        """
        Tests training recipe step.

        Test Case ID: TC-16-01
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
        #assert all(isinstance(metric, torch.nn.Module) for metric in training_recipe.metrics_list), \
        #    "Metrics are not correctly integrated or are not callable modules"
        

        assert isinstance(training_recipe.metrics_list[0], MatchAccuracyMetric)
        assert isinstance(training_recipe.metrics_list[1], FeatureCoverageMetric)