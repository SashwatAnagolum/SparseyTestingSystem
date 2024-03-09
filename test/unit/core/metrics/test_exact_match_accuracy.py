import pytest
import torch

from torch import tensor
from sparsepy.access_objects.models.model import Model
from sparsepy.core.metrics.match_accuracy import MatchAccuracyMetric
from sparsepy.core.model_layers.sparsey_layer import SparseyLayer

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
slay = SparseyLayer(True, 1, 1, 1, 1, 1, 3.0, 1, 1, 2, 2, 4, 0, 28.0, 5.0, 0.5, 1.0)
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


