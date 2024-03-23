import pytest
import torch

from torch import tensor
from sparseypy.access_objects.models.model import Model
from sparseypy.core.metrics.match_accuracy import MatchAccuracyMetric
from sparseypy.core.model_layers.sparsey_layer import SparseyLayer

input_params = [
    torch.tensor([[[[0.]], [[0.]], [[0.]], [[0.]]]]),
    torch.tensor([[[[0.]], [[0.]], [[0.]], [[1.]]]]),
    torch.tensor([[[[0.]], [[0.]], [[1.]], [[0.]]]]),
    torch.tensor([[[[0.]], [[0.]], [[1.]], [[1.]]]]),
    torch.tensor([[[[0.]], [[1.]], [[0.]], [[0.]]]]),
    torch.tensor([[[[0.]], [[1.]], [[0.]], [[1.]]]]),
    torch.tensor([[[[0.]], [[1.]], [[1.]], [[0.]]]]),
    torch.tensor([[[[0.]], [[1.]], [[1.]], [[1.]]]]),
    torch.tensor([[[[1.]], [[0.]], [[0.]], [[0.]]]]),
    torch.tensor([[[[1.]], [[0.]], [[0.]], [[1.]]]]),
    torch.tensor([[[[1.]], [[0.]], [[1.]], [[0.]]]]),
    torch.tensor([[[[1.]], [[0.]], [[1.]], [[1.]]]]),
    torch.tensor([[[[1.]], [[1.]], [[0.]], [[0.]]]]),
    torch.tensor([[[[1.]], [[1.]], [[0.]], [[1.]]]]),
    torch.tensor([[[[1.]], [[1.]], [[1.]], [[0.]]]]),
    torch.tensor([[[[1.]], [[1.]], [[1.]], [[1.]]]])
]

#create 1 layer, 1 mac, 1cm, 1 neuron model, expecting 1x1 input tensor
m = Model()
slay = SparseyLayer(
    autosize_grid=True,  # Assuming this is what True was meant to indicate
    grid_layout="rectangular",  # Assuming a layout, since it's not specified in your snippet
    num_macs=1,
    num_cms_per_mac=1,
    num_neurons_per_cm=1,
    mac_grid_num_rows=1,
    mac_grid_num_cols=1,
    mac_receptive_field_radius=3.0,
    prev_layer_num_cms_per_mac=1,
    prev_layer_num_neurons_per_cm=1,
    prev_layer_mac_grid_num_rows=2,
    prev_layer_mac_grid_num_cols=2,
    prev_layer_num_macs=4,
    prev_layer_grid_layout="rectangular",
    layer_index=0, 
    sigmoid_phi=5.0,
    sigmoid_lambda=28.0,
    saturation_threshold=0.5,
    permanence=1.0, 
    activation_threshold_min=0.2, 
    activation_threshold_max=1.0, 
    min_familiarity=0.2, 
    sigmoid_chi=1.5 
)
m.add_layer(slay)
emam = MatchAccuracyMetric(m)


@pytest.mark.parametrize('input', input_params)
def test_exact_match(input):
    output1 = m(input)

    metric_result = emam.compute(m, input, output1, True)

    output2 = m(input)
    metric_result = emam.compute(m, input, output2, False)

    print(metric_result)
    if torch.equal(output1, output2):
        assert metric_result == [[1.0]]
    else:
        assert metric_result == [[0.0]]


