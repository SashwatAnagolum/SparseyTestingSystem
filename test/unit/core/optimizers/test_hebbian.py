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
            num_macs=1,
            num_cms_per_mac=5,
            num_neurons_per_cm=5,
            mac_grid_num_rows=1,
            mac_grid_num_cols=1,
            prev_layer_num_macs=1,
            mac_receptive_field_size=1.5,
            prev_layer_num_cms_per_mac=10,
            prev_layer_num_neurons_per_cm=10,
            prev_layer_mac_grid_num_rows=1,
            prev_layer_mac_grid_num_cols=1,
            prev_layer_grid_layout="rect",
            layer_index=2,
            sigmoid_phi=5.0,
            sigmoid_lambda=28.0,
            saturation_threshold=0.1,
            permanence_steps=10,
            permanence_convexity=5.0,
            activation_threshold_max=1.0,
            activation_threshold_min=0.2,
            min_familiarity=0.2,
            sigmoid_chi=2.5,
            device=torch.device("cpu")           
        )

        simple_model.add_layer(sparsey_layer)

        return simple_model


    @pytest.fixture
    def simple_model_2(self):
        """
        Returns a sample SparseyLayer object to perform
        tests with.
        """
        simple_model_2 = Model(device='cpu')
        sparsey_layer_2 = SparseyLayer(
            autosize_grid=False,
            grid_layout="rect",
            num_macs=1,
            num_cms_per_mac=5,
            num_neurons_per_cm=5,
            mac_grid_num_rows=1,
            mac_grid_num_cols=1,
            prev_layer_num_macs=1,
            mac_receptive_field_size=1.5,
            prev_layer_num_cms_per_mac=10,
            prev_layer_num_neurons_per_cm=10,
            prev_layer_mac_grid_num_rows=1,
            prev_layer_mac_grid_num_cols=1,
            prev_layer_grid_layout="rect",
            layer_index=2,
            sigmoid_phi=5.0,
            sigmoid_lambda=28.0,
            saturation_threshold=0.1,
            permanence_steps=int(1e8),
            permanence_convexity=0.0,
            activation_threshold_max=1.0,
            activation_threshold_min=0.2,
            min_familiarity=0.2,
            sigmoid_chi=2.5,
            device=torch.device("cpu")           
        )

        simple_model_2.add_layer(sparsey_layer_2)

        return simple_model_2


    def test_weight_updates(self, simple_model) -> None:
        """
        TC-02-01: Tests the weight updates performed by the Hebbian optimizer to ensure it correctly captures
        pre-post correlations for each weight in the Sparsey model and updates them accordingly.
        """
        #Initialize optimizer and hook
        hook = LayerIOHook(simple_model)
        optimizer = HebbianOptimizer(simple_model, torch.device('cpu'))

        input_tensor = torch.zeros((1, 1, 100), dtype=torch.float32)
        input_tensor[0, 0, [1, 2, 4, 8, 16, 32, 64]] = 1.0
        output = simple_model(input_tensor)

        optimizer.step()

        active_neurons = torch.argwhere(output)[:, -1]
        layers, _, _ = hook.get_layer_io()

        assert (
            torch.sum(layers[0].weights).item() == 7 * 5
        ) and (
            torch.sum(
                layers[0].weights[:, [1, 2, 4, 8, 16, 32, 64]]
            ).item() == 7 * 5
        ) and (
            torch.sum(
                layers[0].weights[:, :, active_neurons]
            ).item() == 7 * 5
        )


    def test_weight_freezing(self, simple_model_2) -> None:
        """
        TC-02-02: Tests the weight freezing logic in the Hebbian optimizer,
        which should activate when the fraction of a neuron's incoming
        active weights crosses a user-defined threshold,
        freezing all further weight updates.
        """
        #Initialize optimizer and hook
        optimizer = HebbianOptimizer(simple_model_2, torch.device('cpu'))
        layer = simple_model_2.get_submodule(f'Layer_0')
        weights = layer.weights

        input_tensor = torch.zeros((1, 1, 100), dtype=torch.float32)
        input_tensor[:, :, 0:10] = 1

        output = simple_model_2(input_tensor)
        optimizer.step()
        active_neurons = torch.argwhere(output)[:, -1]
        overlapping_neurons = None

        num_trials = 1

        while True:
            input_tensor = torch.zeros((1, 1, 100), dtype=torch.float32)
            input_tensor[:, :, 10 * num_trials - 5:10 * num_trials + 5] = 1
            num_trials = (num_trials + 1) % 10

            output = simple_model_2(input_tensor)

            curr_active_neurons = torch.argwhere(output)[:, -1]
            has_overlap = torch.eq(
                active_neurons, curr_active_neurons.unsqueeze(-1)
            )

            if torch.sum(has_overlap):
                overlapping_neurons = active_neurons[
                    torch.argwhere(has_overlap.sum(1))
                ]

                break

        assert torch.all(
            torch.le(
                torch.sum(weights[:, :, overlapping_neurons], dim=(0, 1)),
                layer.prev_layer_output_shape[1] * layer.saturation_threshold
            )
        )


    def test_weight_permanence(self, simple_model) -> None:
        """
        TC-02-03: Tests the secondary weight updates to implement the
        permanence feature of weights in Sparsey models, ensuring weights
        not set during the current frame decay according to an
        exponential schedule.
        """
        #Initialize optimizer and hook
        optimizer = HebbianOptimizer(simple_model, torch.device('cpu'))
        layer = simple_model.get_submodule(f'Layer_0')
        weights = layer.weights
        steps = layer.permanence_steps
        convexity = layer.permanence_convexity

        input_tensor = torch.zeros((1, 1, 100), dtype=torch.float32)
        input_tensor[:, :, 0:10] = 1
        output = simple_model(input_tensor)
        optimizer.step()

        active_neurons = torch.argwhere(output)[:, -1]
        active_weight_indices = [i for i in range(10)]
        input_tensor[:, :, :] = 0.0

        for i in range(steps + 5):
            if i < steps:
                expected_weight_value = torch.tensor(
                    (1.0 + (convexity / steps)) / (
                        1.0 + (
                            convexity / (
                                steps - i
                            )
                        )
                    )
                )
            else:
                expected_weight_value = torch.zeros(
                    (), dtype=torch.float32
                )

            assert torch.allclose(
                weights[0, active_weight_indices][:, active_neurons],
                expected_weight_value,
                atol=1e-5
            )

            output = simple_model(input_tensor)
            optimizer.step()
