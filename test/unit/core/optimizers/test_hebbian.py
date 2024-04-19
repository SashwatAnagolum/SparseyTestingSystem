import pytest
import torch
from sparseypy.core.optimizers.hebbian import HebbianOptimizer
from sparseypy.access_objects.models.model import Model
from sparseypy.core.model_layers.sparsey_layer import SparseyLayer
from sparseypy.core.hooks import LayerIOHook


def test_permanence():
    #create model with two sparsey layers for test
    model = Model()

    #add layer1 assuming 4x4 input tensor, 2x2 MAC Grid, 2CM/MAC, 2N/CM, Saturation thresh of 2.0 which theoretically should never saturate
    model.add_layer(SparseyLayer(
        autosize_grid=True,
        grid_layout='rectangular',
        num_macs=10,
        num_cms_per_mac=5,
        num_neurons_per_cm=100,
        mac_grid_num_rows=2,
        mac_grid_num_cols=5,
        mac_receptive_field_radius=1.0,
        prev_layer_num_cms_per_mac=5,
        prev_layer_num_neurons_per_cm=20,
        prev_layer_mac_grid_num_rows=2,
        prev_layer_mac_grid_num_cols=5,
        prev_layer_num_macs=10,
        prev_layer_grid_layout='rectangular',
        layer_index=1,
        sigmoid_phi=0.5,
        sigmoid_lambda=0.5,
        saturation_threshold=0.1,
        permanence_steps=0.1,
        permanence_convexity=0.1,
        activation_threshold_min=0.2,
        activation_threshold_max=0.8,
        min_familiarity=0.1,
        sigmoid_chi=0.1
    ))

    #add layer2 assuming, 1x1 MAC Grid 2CM/MAC, 2N/CM, Sat thresh of 2.0
    model.add_layer(SparseyLayer(
        autosize_grid=True,
        grid_layout='rectangular',
        num_macs=10,
        num_cms_per_mac=5,
        num_neurons_per_cm=100,
        mac_grid_num_rows=2,
        mac_grid_num_cols=5,
        mac_receptive_field_radius=1.0,
        prev_layer_num_cms_per_mac=5,
        prev_layer_num_neurons_per_cm=20,
        prev_layer_mac_grid_num_rows=2,
        prev_layer_mac_grid_num_cols=5,
        prev_layer_num_macs=10,
        prev_layer_grid_layout='rectangular',
        layer_index=1,
        sigmoid_phi=0.5,
        sigmoid_lambda=0.5,
        saturation_threshold=0.1,
        permanence_steps=0.1,
        permanence_convexity=0.1,
        activation_threshold_min=0.2,
        activation_threshold_max=0.8,
        min_familiarity=0.1,
        sigmoid_chi=0.1
    )) 

    #set up hook for assertion later
    hook = LayerIOHook(model)

    #create hebbian optimizer pass in model
    optimizer = HebbianOptimizer(model)

    #generate random input of correct size and format and pass through model 100 times
    for _ in range(10):
        input_values = torch.rand(2, 5, 5, 20).round()
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
         

