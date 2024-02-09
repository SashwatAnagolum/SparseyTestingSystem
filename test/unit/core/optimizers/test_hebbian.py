import pytest
import torch
from sparsepy.core.optimizers.hebbian import HebbianOptimizer
from sparsepy.access_objects.models.model import Model

'''
test_cases = [
    # Weights, Threshold, Expected Mask
    (torch.tensor([[0.2, 0.4, 0.6], [0.8, 1.0, 0.9]], dtype=torch.float32), 0.7, torch.tensor([[1., 1., 1.], [0., 0., 0.]], dtype=torch.float32)),
    (torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float32), 0.7, torch.tensor([[1., 1., 1.], [1., 1., 1.]], dtype=torch.float32)),
    (torch.tensor([[0.8, 0.9, 1.0], [0.9, 1.0, 1.1]], dtype=torch.float32), 0.7, torch.tensor([[0., 0., 0.], [0., 0., 0.]], dtype=torch.float32)),
    (torch.tensor([[0.6, 0.7, 0.8], [0.5, 0.6, 0.7]], dtype=torch.float32), 0.65, torch.tensor([[1., 0., 0.], [1., 1., 0.]], dtype=torch.float32)),
    (torch.tensor([[0.6, 0.7, 0.8], [0.5, 0.65, 0.8]], dtype=torch.float32), 0.7, torch.tensor([[1., 0., 0.], [1., 1., 1.]], dtype=torch.float32)),
    (torch.tensor([[0.2, 0.4, 0.5], [0.6, 0.8, 0.9]], dtype=torch.float32), 0.55, torch.tensor([[1., 1., 1.], [0., 0., 0.]], dtype=torch.float32)),
    (torch.tensor([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]], dtype=torch.float32), 0.25, torch.tensor([[1., 1., 0.], [1., 0., 0.]], dtype=torch.float32)),
    (torch.tensor([[0.5, 0.6, 0.7], [0.6, 0.7, 0.8]], dtype=torch.float32), 0.65, torch.tensor([[1., 1., 0.], [1., 0., 0.]], dtype=torch.float32)),
    (torch.tensor([[0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=torch.float32), 0.75, torch.tensor([[1., 1., 1.], [1., 0., 0.]], dtype=torch.float32)),
    (torch.tensor([[0.3, 0.4, 0.5], [0.5, 0.6, 0.7]], dtype=torch.float32), 0.4, torch.tensor([[0., 0., 0.], [0., 0., 0.]], dtype=torch.float32)),

    # Edge Case: All weights are exactly at the threshold
    (torch.tensor([[0.7, 0.7, 0.7], [0.7, 0.7, 0.7]], dtype=torch.float32), 0.7, torch.tensor([[0., 0., 0.], [0., 0., 0.]], dtype=torch.float32)),
    
    # Edge Case: Threshold is zero
    (torch.tensor([[0.2, 0.4, 0.6], [0.8, 1.0, 0.9]], dtype=torch.float32), 0.0, torch.tensor([[0., 0., 0.], [0., 0., 0.]], dtype=torch.float32)),

    # Edge Case: Threshold is very high
    (torch.tensor([[0.2, 0.4, 0.6], [0.8, 1.0, 0.9]], dtype=torch.float32), 0.95, torch.tensor([[1., 1., 1.], [1., 1., 1.]], dtype=torch.float32)),
    
    # Edge Case: Negative weights and threshold
    (torch.tensor([[-0.2, -0.4, -0.6], [-0.8, -1.0, -0.9]], dtype=torch.float32), -0.7, torch.tensor([[0., 0., 0.], [0., 0., 0.]], dtype=torch.float32)),
    
    # Edge Case: Mixed positive and negative weights
    (torch.tensor([[0.2, -0.4, 0.6], [-0.8, 1.0, -0.9]], dtype=torch.float32), 0.3, torch.tensor([[1., 0., 1.], [0., 1., 0.]], dtype=torch.float32)),

    # Edge Case: Weights are all the same
    (torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], dtype=torch.float32), 0.5, torch.tensor([[1., 1., 1.], [1., 1., 1.]], dtype=torch.float32)),
    
    # Edge Case: Threshold exactly in the middle of the range
    (torch.tensor([[0.25, 0.5, 0.75], [0.1, 0.4, 0.7]], dtype=torch.float32), 0.5, torch.tensor([[1., 1., 1.], [1., 1., 0.]], dtype=torch.float32)),

    # Edge Case: Extremely small and large values
    (torch.tensor([[1e-5, 1e-3, 1e-2], [1e1, 1e2, 1e3]], dtype=torch.float32), 1, torch.tensor([[1., 1., 1.], [0., 0., 0.]], dtype=torch.float32)),
    
    # Edge Case: Zero weights
    (torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32), 0.5, torch.tensor([[1., 1., 1.], [1., 1., 1.]], dtype=torch.float32)),
    
    # Edge Case: One weight per row
    (torch.tensor([[0.2], [0.8]], dtype=torch.float32), 0.5, torch.tensor([[1.], [0.]], dtype=torch.float32))
]

@pytest.mark.parametrize("mock_weights, mock_threshold, expected_mask", test_cases)
def test_calculate_freezing_mask(mock_weights, mock_threshold, expected_mask):

    #Instantiate HebbianOptimizer with arbitrary model
    m = Model()
    model_layer = torch.nn.Linear(5,2)
    m.add_layer(model_layer)
    optimizer = HebbianOptimizer(m)

    # Mock the saturation thresholds list iwth our mock threshold
    optimizer.saturation_thresholds = [mock_threshold]

    # Call the method under test
    mask = optimizer.calculate_freezing_mask(mock_weights, layer_index=0)

     # Check if the calculated mask matches the expected mask
    assert torch.equal(mask, expected_mask), (
        f"The freezing mask calculated does not match the expected mask.\n"
        f"Expected Mask:\n{expected_mask}\n"
        f"Actual Mask:\n{mask}"
    )
    '''


def test_mean_inputs_increasing():
    model = Model()
    # Assuming model has layers and parameters
    optimizer = HebbianOptimizer(model)
    
    # Simulate multiple steps
    #This needs to change so that the steps have a real model and present real data
    previous_mean_input = None
    for step in range(5):  # Example: 5 steps
        optimizer.step()
        
        # Assuming we have a way to access the mean inputs after each step
        current_mean_input = ...  # Fetch the current mean inputs
        if previous_mean_input is not None:
            assert current_mean_input > previous_mean_input, "Mean inputs did not increase after step {}".format(step)
        previous_mean_input = current_mean_input

def test_neuron_freezing_after_threshold():
    model = Model()
    # Assuming model has layers and parameters
    optimizer = HebbianOptimizer(model)

    # Simulate steps until the mean inputs surpass the threshold
    while True:
        optimizer.step()
        
        # Fetch current mean inputs and the updateable mask
        current_mean_input = ...  # Get current mean inputs
        updateable_mask = ...  # Get current updateable mask

        # Check if mean inputs have surpassed the threshold
        if (current_mean_input > optimizer.saturation_thresholds[0]).any():
            # Verify that the corresponding positions in the updateable mask are frozen (0)
            assert (updateable_mask[current_mean_input > optimizer.saturation_thresholds[0]] == 0).all(), \
                   "Neurons with mean inputs above threshold are not correctly frozen."
            break
