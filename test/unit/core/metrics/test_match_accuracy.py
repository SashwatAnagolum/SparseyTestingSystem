import pytest
import torch

from torch import tensor
from sparseypy.access_objects.models.model import Model
from sparseypy.core.metrics.match_accuracy import MatchAccuracyMetric
from sparseypy.core.metrics.match_accuracy import MatchAccuracyMetric
from sparseypy.core.model_layers.sparsey_layer import SparseyLayer
#this should be more aimed at testing the ability to determine the closest match relative to stored inputs

def test_approximate_match():
    """
    Tests the ability of MatchAccuracyMetric to discern approximate matches among stored inputs,
    focusing on its functionality in recognizing the closest match relative to stored inputs.

    Test case ID: TC-05-03: Test match accuracy metric.
    """
    
    #create 1 layer, 1 mac, 1cm, 1 neuron model, expecting 2x2 input tensor
    m = Model(torch.device("cpu"))
    slay = SparseyLayer(
    autosize_grid=True,  
    grid_layout="rectangular",
    num_macs=1,
    num_cms_per_mac=1,
    num_neurons_per_cm=1,
    mac_grid_num_rows=1,
    mac_grid_num_cols=1,
    mac_receptive_field_size=3.0,
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
    permanence_steps=10, 
    permanence_convexity=0.5,
    activation_threshold_min=0.2, 
    activation_threshold_max=1.0, 
    min_familiarity=0.2, 
    sigmoid_chi=1.5, 
    device=torch.device("cpu")
)
    
    #SparseyLayer(True, 1, 1, 1, 1, 1, 3.0, 1, 1, 2, 2, 4, 0, 28.0, 5.0, 0.5, 1.0)
    m.add_layer(slay)
    amam = MatchAccuracyMetric(m, torch.device("cpu"))

    #three inputs that are very different from one target input
    diff_in_1 = torch.tensor([[
        [1.], [1.], [1.], [1.]
    ]])
    diff_in_2 = torch.tensor([[
        [1.], [1.], [1.], [0.]
    ]])
    diff_in_3 = torch.tensor([[
        [1.], [1.], [0.], [1.]
    ]])

    #perform training run on the three of these inputs, each with the same model with empty outputs because it has not been passed data
    #output = torch.tensor([])
    output = m(diff_in_1)#model is passed data, an now has layer_ouputs on hooks
    amam.compute(m, diff_in_1, output, True)
    output = m(diff_in_2)
    amam.compute(m, diff_in_2, output, True)
    output = m(diff_in_3)
    amam.compute(m, diff_in_3, output, True)
    


    #one input with another slightly altered version of itself
    norm_in = torch.tensor([
        [[0.], [0.], [0.], [0.]]
    ])
    perm_in = torch.tensor([
        [[0.], [0.], [0.], [1.]]
    ])


    #perform training run on normal input, perform evaluation run on permuted input
    output = m(norm_in)#model is passed data, an now has layer_ouputs on hooks
    amam.compute(m, norm_in, output, True)
    output = m(perm_in)
    metric_result = amam.compute(m, perm_in, output, False)

    assert torch.stack([t.squeeze() for t in metric_result]) == tensor([1.])
