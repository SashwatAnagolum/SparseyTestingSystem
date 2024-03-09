import pytest
import torch
import logging

from sparsepy.cli.config_validation.validate_config import (
    validate_config, get_config_info
)
from sparsepy.access_objects.models.model_builder import ModelBuilder
from sparsepy.access_objects.training_recipes.training_recipe_builder import TrainingRecipeBuilder
from sparsepy.access_objects.models.model import Model
from sparsepy.core.optimizers.hebbian import HebbianOptimizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@pytest.mark.parametrize(
    "model_config_path, dataset_config_path, preprocessing_config_path, trainer_config_path",
    [
        # Each tuple corresponds to a set of paths for a single test run
        (
            r"test\system\weight_freezing\test_cases\tc1\model.yaml",
            r"test\system\weight_freezing\test_cases\tc1\dataset.yaml",
            r"test\system\weight_freezing\test_cases\tc1\preprocessing.yaml",
            r"test\system\weight_freezing\test_cases\tc1\trainer.yaml"
        ),
        (
            r"test\system\weight_freezing\test_cases\tc2\model.yaml",
            r"test\system\weight_freezing\test_cases\tc2\dataset.yaml",
            r"test\system\weight_freezing\test_cases\tc2\preprocessing.yaml",
            r"test\system\weight_freezing\test_cases\tc2\trainer.yaml"
        ),
        (
            r"test\system\weight_freezing\test_cases\tc3\model.yaml",
            r"test\system\weight_freezing\test_cases\tc3\dataset.yaml",
            r"test\system\weight_freezing\test_cases\tc3\preprocessing.yaml",
            r"test\system\weight_freezing\test_cases\tc3\trainer.yaml"
        ),
        (
            r"test\system\weight_freezing\test_cases\tc4\model.yaml",
            r"test\system\weight_freezing\test_cases\tc4\dataset.yaml",
            r"test\system\weight_freezing\test_cases\tc4\preprocessing.yaml",
            r"test\system\weight_freezing\test_cases\tc4\trainer.yaml"
        ),
        (
            r"test\system\weight_freezing\test_cases\tc5\model.yaml",
            r"test\system\weight_freezing\test_cases\tc5\dataset.yaml",
            r"test\system\weight_freezing\test_cases\tc5\preprocessing.yaml",
            r"test\system\weight_freezing\test_cases\tc5\trainer.yaml"
        ),
        (
            r"test\system\weight_freezing\test_cases\tc6\model.yaml",
            r"test\system\weight_freezing\test_cases\tc6\dataset.yaml",
            r"test\system\weight_freezing\test_cases\tc6\preprocessing.yaml",
            r"test\system\weight_freezing\test_cases\tc6\trainer.yaml"
        ),
        (
            r"test\system\weight_freezing\test_cases\tc7\model.yaml",
            r"test\system\weight_freezing\test_cases\tc7\dataset.yaml",
            r"test\system\weight_freezing\test_cases\tc7\preprocessing.yaml",
            r"test\system\weight_freezing\test_cases\tc7\trainer.yaml"
        ),
        (
            r"test\system\weight_freezing\test_cases\tc8\model.yaml",
            r"test\system\weight_freezing\test_cases\tc8\dataset.yaml",
            r"test\system\weight_freezing\test_cases\tc8\preprocessing.yaml",
            r"test\system\weight_freezing\test_cases\tc8\trainer.yaml"
        ),
        (
            r"test\system\weight_freezing\test_cases\tc9\model.yaml",
            r"test\system\weight_freezing\test_cases\tc9\dataset.yaml",
            r"test\system\weight_freezing\test_cases\tc9\preprocessing.yaml",
            r"test\system\weight_freezing\test_cases\tc9\trainer.yaml"
        ),
        (
            r"test\system\weight_freezing\test_cases\tc10\model.yaml",
            r"test\system\weight_freezing\test_cases\tc10\dataset.yaml",
            r"test\system\weight_freezing\test_cases\tc10\preprocessing.yaml",
            r"test\system\weight_freezing\test_cases\tc10\trainer.yaml"
        )
        # Add more tuples as needed for additional test runs
    ]
)
def test_train_model(model_config_path, dataset_config_path, preprocessing_config_path, trainer_config_path):

    logging.info(f"Test Params:\n   model: {model_config_path}\n   dataset: {dataset_config_path}\n   preprocessing: {preprocessing_config_path}\n   trainer: {trainer_config_path}")

    model_config = get_config_info(model_config_path)
    model_config, is_valid = validate_config(
        model_config, 'model', 'sparsey'
    )
    model = ModelBuilder.build_model(model_config)


    dataset_config = get_config_info(dataset_config_path)
    dataset_config, is_valid = validate_config(
        dataset_config, 'dataset', dataset_config['dataset_type']
    )

    preprocessing_config = get_config_info(preprocessing_config_path)

    trainer_config = get_config_info(trainer_config_path)
    trainer_config, is_valid = validate_config(
        trainer_config, 'training_recipe', 'sparsey'
    )

    trainer = TrainingRecipeBuilder.build_training_recipe(
        model, dataset_config, preprocessing_config,
        trainer_config
    )

    saturation_thresholds = trainer.optimizer.saturation_thresholds

    is_epoch_done = False
    model.train()

    previous_means = None    

    while not is_epoch_done:
        output, is_epoch_done = trainer.step(training=True)
        #assert that the weights are strictly increasing
        #assert that once the mean weights hit the threshold for their layer, freezing occurs and inputs to that neuron stop updating
        current_macs, inputs, outputs = trainer.optimizer.hook.get_layer_io()
        #calculate current means
        #current_means = torch.tensor([
            #torch.mean(mac.parameters()[0].data, dim=1)
            #for mac in current_macs
            #])
        
        current_means = []

        #new code
        # Loop to calculate current means for each MAC
    for current_layer in current_macs:
        for mac in current_layer:
            # This assumes mac.parameters() yields at least one parameter tensor
            # and that you're interested in the first parameter's data for mean calculation
            params = next(mac.parameters(), None)
            if params is not None:
                mean_input = torch.mean(params.data, dim=1)
                current_means.append(mean_input)
            else:
                # Handle the case where a MAC has no parameters, if possible
                current_means.append(torch.tensor([]))  # Placeholder for MACs without parameters

    # Check against previous_means, if they exist
    if previous_means is not None:
        for i, (prev_mean, curr_mean) in enumerate(zip(previous_means, current_means)):
            # Access the correct threshold based on the MAC's layer index
            threshold = saturation_thresholds[current_macs[i].layer_index]
            
            # Ensure prev_mean has elements before comparing
            if prev_mean.numel() > 0:
                frozen_neurons = prev_mean > threshold
                assert torch.all(curr_mean[frozen_neurons] == prev_mean[frozen_neurons]), (
                    f"Neurons that surpassed the threshold were updated for MAC {i}. "
                    f"Current frozen values: {curr_mean[frozen_neurons]}, Previous frozen values: {prev_mean[frozen_neurons]}"
                )
                # For neurons not frozen, check if the current mean is greater or equal
                not_frozen = ~frozen_neurons
                if curr_mean[not_frozen].numel() > 0:  # Check if there are any not frozen neurons
                    assert torch.all(curr_mean[not_frozen] >= prev_mean[not_frozen]), (
                        f"Non-frozen neurons did not increase or stay the same for MAC {i}. "
                        f"Previous: {prev_mean[not_frozen]}, Current: {curr_mean[not_frozen]}"
                    )
            
        previous_means = [mean.clone().detach() for mean in current_means]

        
        





