import pytest
import torch

from torch import tensor
from sparsepy.access_objects.models.model import Model
from sparsepy.core.metrics.match_accuracy import MatchAccuracyMetric
from sparsepy.core.metrics.exact_match_accuracy import ExactMatchAccuracyMetric
from sparsepy.core.model_layers.sparsey_layer import SparseyLayer
#this should be more aimed at testing the ability to determine the closest match relative to stored inputs

def test_approximate_match():
    #create 1 layer, 1 mac, 1cm, 1 neuron model, expecting 2x2 input tensor
    m = Model()
    slay = SparseyLayer(True, 1, 1, 1, 1, 1, 3.0, 1, 1, 2, 2, 4, 0, 28.0, 5.0, 0.5, 1.0)
    m.add_layer(slay)
    amam = MatchAccuracyMetric(m)

    #three inputs that are very different from one target input
    diff_in_1 = torch.tensor([[[[1.]], [[1.]], [[1.]], [[1.]]]])
    diff_in_2 = torch.tensor([[[[1.]], [[1.]], [[1.]], [[0.]]]])
    diff_in_3 = torch.tensor([[[[1.]], [[1.]], [[0.]], [[1.]]]])

    #perform training run on the three of these inputs, each with the same model with empty outputs because it has not been passed data
    output = torch.tensor([])
    amam.compute(m, diff_in_1, output, True)
    amam.compute(m, diff_in_2, output, True)
    amam.compute(m, diff_in_3, output, True)
    


    #one input with another slightly altered version of itself
    norm_in = torch.tensor([[[[0.]], [[0.]], [[0.]], [[0.]]]])
    perm_in = torch.tensor([[[[0.]], [[0.]], [[0.]], [[1.]]]])


    #perform training run on normal input, perform evaluation run on permuted input
    output = m(norm_in)#model is passed data, an now has layer_ouputs on hooks
    amam.compute(m, norm_in, output, True)
    output = m(perm_in)
    metric_result = amam.compute(m, perm_in, output, False)

    assert metric_result == [[1.0]]
