import torch
import pytest
from unittest.mock import Mock
from sparseypy.core.metrics.feature_coverage import FeatureCoverageMetric
from sparseypy.access_objects.models.model import Model
from sparseypy.core.model_layers.sparsey_layer import SparseyLayer

def test_feature_coverage_compute():
    model = Model(torch.device("cpu"))
    layer = SparseyLayer(
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
            permanence_steps=0.1, 
            permanence_convexity=0.1, 
            activation_threshold_min=0.2, 
            activation_threshold_max=1.0, 
            min_familiarity=0.2, 
            sigmoid_chi=1.5,
            device=torch.device("cpu")
        )
    
    model.add_layer(layer)
    metric = FeatureCoverageMetric(model, torch.device("cpu"), reduction='mean')
    last_batch = torch.rand((10, 10))
    labels = torch.rand((10, 5))

    result = metric.compute(model, last_batch, labels, training=True)

    # Verify the result is a torch.Tensor
    assert isinstance(result, torch.Tensor), "Compute method should return a torch.Tensor"
    # TODO Add functionality testing typical Sparsey inputs