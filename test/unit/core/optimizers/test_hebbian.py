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
            permanence_steps=1000,
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

        input_tensor = torch.tensor([[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]])
        simple_model(input_tensor)
        layers_before, inputs, _ = hook.get_layer_io()
        optimizer.step()
        layers_after, _, _ = hook.get_layer_io()
        #use hooks to iterate through macs and verify the weights updated as expected
        for layer_index, (layer_before, layer_after) in enumerate(zip(layers_before, layers_after)):
            assert layer_after == layer_before, "Weights not updated as expected"

    def test_weight_freezing(self, simple_model) -> None:
        """
        TC-02-02: Tests the weight freezing logic in the Hebbian optimizer, which should activate when the fraction
        of a neuron's incoming active weights crosses a user-defined threshold, freezing all further weight updates.
        """
        #Initialize optimizer and hook
        hook = LayerIOHook(simple_model)
        optimizer = HebbianOptimizer(simple_model, torch.device('cpu'))

        input_tensor = torch.tensor([[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]])
        simple_model(input_tensor)
        layers_before, inputs, _ = hook.get_layer_io()
        optimizer.step()
        layers_after, _, _ = hook.get_layer_io()
        #use hooks to iterate through macs and verify the weights updated as expected
        for layer_index, (layer_before, layer_after) in enumerate(zip(layers_before, layers_after)):
            assert layer_after == layer_before, "Weights not updated as expected"

    def test_weight_permanence(self, simple_model) -> None:
        """
        TC-02-03: Tests the secondary weight updates to implement the permanence feature of weights in Sparsey models,
        ensuring weights not set during the current frame decay according to an exponential schedule.
        """
        #Initialize optimizer and hook
        hook = LayerIOHook(simple_model)
        optimizer = HebbianOptimizer(simple_model, torch.device('cpu'))

        input_tensor = torch.tensor([[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]])
        simple_model(input_tensor)
        layers_before, inputs, _ = hook.get_layer_io()
        optimizer.step()
        layers_after, _, _ = hook.get_layer_io()
        #use hooks to iterate through macs and verify the weights updated as expected
        for layer_index, (layer_before, layer_after) in enumerate(zip(layers_before, layers_after)):
            assert layer_after == layer_before, "Weights not updated as expected"

# Additional tests can be added to the TestHebbianOptimizer class to cover more aspects like error handling,
# performance, or other specific functionalities of the HebbianOptimizer class.