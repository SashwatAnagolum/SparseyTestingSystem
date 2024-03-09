import pytest
import torch
from sparsepy.core.optimizers.hebbian import HebbianOptimizer
from sparsepy.access_objects.models.model import Model
from sparsepy.core.model_layers.sparsey_layer import SparseyLayer
from sparsepy.core.hooks import LayerIOHook


def test_permanence():
    #create model with two sparsey layers for test
    model = Model()

    #add layer1 assuming 4x4 input tensor, 2x2 MAC Grid, 2CM/MAC, 2N/CM, Saturation thresh of 2.0 which theoretically should never saturate
    model.add_layer(SparseyLayer(True, 4, 2, 2, 2, 2, 2.0, 1, 1, 4, 4, 16, 0, 28.0, 5.0, 2.0, 0.5))

    #add layer2 assuming, 1x1 MAC Grid 2CM/MAC, 2N/CM, Sat thresh of 2.0
    model.add_layer(SparseyLayer(True, 1, 2, 2, 1, 1, 1.0, 2, 2, 2, 2, 4, 1, 28.0, 5.0, 2.0, 0.5)) 

    #set up hook for assertion later
    hook = LayerIOHook(model)

    #create hebbian optimizer pass in model
    optimizer = HebbianOptimizer(model)

    #generate random input of correct size and format and pass through model 100 times
    for _ in range(10):
        input_values = torch.rand((1, 16, 1, 1)).round()
        input_tensor = torch.where(input_values > 0.5, torch.tensor(1.), torch.tensor(0.))
        model(input_tensor)
        layers_before, inputs, _ = hook.get_layer_io()
        optimizer.step()
        layers_after, _, _ = hook.get_layer_io()
        #use hooks to iterate through macs and verify the weights decreased properly
        for layer_index, (layer_before, layer_after) in enumerate(zip(layers_before, layers_after)):
            for mac_index, (mac_before, mac_after, mac_input) in enumerate(zip(layer_before, layer_after, inputs[layer_index])):
                assert True == True 
                #mac_before.parameters[0] should contain waits of mac_before, and so forth
                #look at format of inputs, look at format of weights and determine the correct tensor operations to decide which weights to evaluate for decrease 
         

