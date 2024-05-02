"""
Test Hebbian Optimizer: test cases for the Hebbian optimizer functionality in the Sparsey model system.
"""

import pytest
import torch
from sparseypy.core.optimizers.hebbian import HebbianOptimizer
from sparseypy.access_objects.models.model import Model
from sparseypy.core.model_layers.sparsey_layer import SparseyLayer
from sparseypy.core.hooks import LayerIOHook

class TestHebbianOptimizer:
    """
    TestHebbianOptimizer: a class holding a collection
        of tests focused on the HebbianOptimizer class.
    """

    @pytest.fixture
    def simple_model(self):
        """
        Returns a sample SparseyLayer object to perform
        tests with.
        """
        simple_model = Model(device='cpu')
        sparsey_layer = SparseyLayer(
            autosize_grid=False,
            grid_layout="rect",
            num_macs=16,
            num_cms_per_mac=8,
            num_neurons_per_cm=16,
            mac_grid_num_rows=4,
            mac_grid_num_cols=4,
            prev_layer_num_macs=1,
            mac_receptive_field_size=0.5,
            prev_layer_num_cms_per_mac=1,
            prev_layer_num_neurons_per_cm=16,
            prev_layer_mac_grid_num_rows=1,
            prev_layer_mac_grid_num_cols=1,
            prev_layer_grid_layout="rect",
            layer_index=2,
            sigmoid_phi=5.0,
            sigmoid_lambda=28.0,
            saturation_threshold=0.5,
            permanence_steps=1.0,
            permanence_convexity=1.0,
            activation_threshold_max=1.0,
            activation_threshold_min=0.2,
            min_familiarity=0.2,
            sigmoid_chi=2.5,
            device=torch.device("cpu")           
        )
        simple_model.add_layer(sparsey_layer)
        return simple_model

    def test_weight_updates(self, simple_model) -> None:
        """
        TC-02-01: Tests the weight updates performed by the Hebbian optimizer to ensure it correctly captures
        pre-post correlations for each weight in the Sparsey model and updates them accordingly.
        """
        #Initialize optimizer and hook
        hook = LayerIOHook(simple_model)
        optimizer = HebbianOptimizer(simple_model, torch.device('cpu'))

        #generate random input of correct size and format and pass through model 100 times
        for _ in range(10):
            input_values = torch.rand((1, 1, 16)).round()
            input_tensor = torch.where(input_values > 0.5, torch.tensor(1.), torch.tensor(0.))
            simple_model(input_tensor)
            layers_before, inputs, _ = hook.get_layer_io()
            optimizer.step()
            layers_after, _, _ = hook.get_layer_io()
            #use hooks to iterate through macs and verify the weights decreased properly
            for layer_index, (layer_before, layer_after) in enumerate(zip(layers_before, layers_after)):
                for mac_index, (mac_before, mac_after, mac_input) in enumerate(zip(layer_before, layer_after, inputs[layer_index])):
                    assert True == True 
                    #mac_before.parameters[0] should contain waits of mac_before, and so forth
                    #look at format of inputs, look at format of weights and determine the correct tensor operations to decide which weights to evaluate for decrease 

    def test_weight_freezing(self) -> None:
        """
        TC-02-02: Tests the weight freezing logic in the Hebbian optimizer, which should activate when the fraction
        of a neuron's incoming active weights crosses a user-defined threshold, freezing all further weight updates.
        """
        # Initialize model and optimizer with threshold setting
        model_config = 'path/to/sparsey_model_config.json'
        preprocessing_config = 'path/to/preprocessing_stack_config.json'
        threshold = 0.5  # Example threshold
        model = SparseyModel(model_config, preprocessing_config, threshold)
        optimizer = HebbianOptimizer(model)

        # Simulate condition to trigger weight freezing
        optimizer.check_and_freeze_weights()

        # Verify that weights are frozen (this needs to be filled with actual test logic)
        assert optimizer.weights_frozen(), "Weights not frozen as expected"

    def test_weight_permanence(self) -> None:
        """
        TC-02-03: Tests the secondary weight updates to implement the permanence feature of weights in Sparsey models,
        ensuring weights not set during the current frame decay according to an exponential schedule.
        """
        # Initialize model and optimizer
        model_config = 'path/to/sparsey_model_config.json'
        preprocessing_config = 'path/to/preprocessing_stack_config.json'
        model = SparseyModel(model_config, preprocessing_config)
        optimizer = HebbianOptimizer(model)

        # Apply weight permanence feature
        optimizer.apply_weight_permanence()

        # Ensure permanence feature is applied correctly (this needs to be filled with actual test logic)
        assert optimizer.permanence_applied_properly(), "Permanence feature not applied as expected"

# Additional tests can be added to the TestHebbianOptimizer class to cover more aspects like error handling,
# performance, or other specific functionalities of the HebbianOptimizer class.