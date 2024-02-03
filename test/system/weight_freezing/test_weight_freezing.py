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

@pytest.mark.parametrize("model_config_path", [
        r"C:\Users\chris\OneDrive\Desktop\SparseyTestingSystem\SparseyTestingSystem\test\system\weight_freezing\models\model1.yaml",
        r"C:\Users\chris\OneDrive\Desktop\SparseyTestingSystem\SparseyTestingSystem\test\system\weight_freezing\models\model2.yaml",
        r"C:\Users\chris\OneDrive\Desktop\SparseyTestingSystem\SparseyTestingSystem\test\system\weight_freezing\models\model3.yaml",
        r"C:\Users\chris\OneDrive\Desktop\SparseyTestingSystem\SparseyTestingSystem\test\system\weight_freezing\models\model4.yaml",
        r"C:\Users\chris\OneDrive\Desktop\SparseyTestingSystem\SparseyTestingSystem\test\system\weight_freezing\models\model5.yaml"
    ])
def test_train_model(model_config_path):
    logging.info(f"Testing with model config: {model_config_path}")
    model_config = get_config_info(model_config_path)
    model_config, is_valid = validate_config(
        model_config, 'model', 'sparsey'
    )
    model = ModelBuilder.build_model(model_config)


    dataset_config = get_config_info(
        r"C:\Users\chris\OneDrive\Desktop\SparseyTestingSystem\SparseyTestingSystem\test\system\weight_freezing\dataset.yaml"
    )
    dataset_config, is_valid = validate_config(
        dataset_config, 'dataset', 'image'
    )
    preprocessing_config = get_config_info(
        r"C:\Users\chris\OneDrive\Desktop\SparseyTestingSystem\SparseyTestingSystem\test\system\weight_freezing\preprocessing.yaml"
    )
    trainer_config = get_config_info(
        r"C:\Users\chris\OneDrive\Desktop\SparseyTestingSystem\SparseyTestingSystem\test\system\weight_freezing\trainer.yaml"
    )
    trainer_config, is_valid = validate_config(
        trainer_config, 'training_recipe', 'sparsey'
    )
    trainer = TrainingRecipeBuilder.build_training_recipe(
        model, dataset_config, preprocessing_config,
        trainer_config
    )
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
        for mac in current_macs:
            for params in list(mac.parameters()):
                mean_input = torch.mean(params.data, dim=1)
                current_means.append(mean_input)
        current_means = torch.stack(current_means)

        if previous_means is not None:
            for i, (prev_mean, curr_mean) in enumerate(zip(previous_means, current_means)):
                assert torch.all(curr_mean >= prev_mean), (
                    f"Means did not increase or stay the same for MAC {i} at step. "
                    f"Previous: {prev_mean.tolist()}, Current: {curr_mean.tolist()}"
                )
                logging.info(f"Mac {i}: Means increased or remained the same as expected")

                ####Need to know which layer I am at here
                #threshold = trainer.optimizer.saturation_thresholds[fuck]
                threshold = 0.5
                frozen_neurons = prev_mean > threshold
                if any(frozen_neurons):
                    assert torch.all(curr_mean[frozen_neurons] == prev_mean[frozen_neurons]), (
                        f"Neurons that surpassed the threshold were updated for MAC {i} at step."
                        f"Current frozen values: {curr_mean[frozen_neurons]} Previous frozen values: {prev_mean[frozen_neurons]}"
                    )
            
        previous_means = current_means.clone().detach()

        
        





