import pytest
import torch

from torch import tensor
from sparsepy.access_objects.models.model import Model
from sparsepy.core.metrics.approximate_match_accuracy import ApproximateMatchAccuracyMetric
from sparsepy.core.metrics.exact_match_accuracy import ExactMatchAccuracyMetric
from sparsepy.core.model_layers.sparsey_layer import SparseyLayer



#create 1 layer, 1 mac, 1cm, 1 neuron model, expecting 1x1 input tensor
m = Model()
slay = SparseyLayer(1, 1, 1, 1, 1, 2.0, 1, 1, 1, 1, 1, 0, 28.0, 5.0, 0.5, 1.0)
m.add_layer(slay)
emam = ExactMatchAccuracyMetric(m)


#create input and pass to model then pass to metric
input = tensor([[[[1.]]]])
output1 = m(input)

metric_result = emam.compute(m, input, output1, True)

output2 = m(input)
metric_result = emam.compute(m, input, output2, False)

print(metric_result)

def test_exact_match():
    if torch.equal(output1, output2):
        assert metric_result == 1.0
    else:
        print("Outputs were not equal.")
