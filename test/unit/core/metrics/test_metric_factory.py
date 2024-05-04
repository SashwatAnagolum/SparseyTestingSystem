# -*- coding: utf-8 -*-

"""
Test Metric Factory: test cases for the MetricFactory class.
"""


import pytest
import torch

from sparseypy.core import metrics
from sparseypy.core.metrics.metric_factory import MetricFactory
from sparseypy.core.metrics.metrics import Metric
from sparseypy.core.metrics.feature_coverage import FeatureCoverageMetric
from sparseypy.access_objects.models.model import Model
from sparseypy.core.model_layers.sparsey_layer import SparseyLayer

class TestMetricFactory:
    """
    TestMetricFactory: a class holding a collection
        of tests focused on the MetricFactory class.
    """
    def test_valid_metric_name(self) -> None:
        """
        Tests whether the MetricFactory correctly loads 
        a class if we provide it with a valid layer name.

        Test case ID: TC-05-01: Test for valid system metric name.
        """
        metric = MetricFactory.get_metric_class('match_accuracy')

        print(metric)

        assert issubclass(metric, Metric)

    def test_valid_metric_feature_coverage(self):
        """
        Test if the MetricFactory correctly loads a FeatureCoverageMetric class.
        Test case ID: TC-05-04: Test for valid feature coverage metric.
        """
        # Simulate a model object since FeatureCoverageMetric expects one
        model = Model(device=torch.device("cpu"))
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
            sigmoid_chi=1.5 ,
            device=torch.device("cpu")
        )
    
        model.add_layer(layer)
        # Attempt to get the FeatureCoverageMetric class from the factory
        metric_class = MetricFactory.get_metric_class('feature_coverage')

        # Verify that the returned class is FeatureCoverageMetric
        assert metric_class == FeatureCoverageMetric, "MetricFactory did not return the expected metric class."

        # Instantiate the metric to verify it can be created successfully
        metric = metric_class(model=model, reduction='mean', device=torch.device('cpu'))

        # Create mock inputs, should use better tensors
        mock_input = torch.rand(1, 4, 1)
        #mock_labels = torch.randint(0, 5, (10,)) 

        mock_labels = model(mock_input) 

        # Call compute to simulate metric computation
        result = metric.compute(model, mock_input, mock_labels, training=True)

        # Basic check to verify that result is a tensor
        assert isinstance(result, torch.Tensor), "The compute method did not return a tensor."


    def test_invalid_metric_name(self) -> None:
        """
        Tests whether the MetricFactory throws an error  
        if we provide it with a invalid system metric name.

        Test case ID: TC-05-05: Test for invalid metric name.
        """
        with pytest.raises(ValueError):
            MetricFactory.get_metric_class('martian_accuracy')

    def test_invalid_pytorch_metric_name(self) -> None:
        """
        Tests whether the MetricFactory throws an error when passed an invalid Pytorch metric.

        Test case ID: TC-05-07: Test for invalid pytorch metric name.
        """
        with pytest.raises(ValueError):
            MetricFactory.get_metric_class('PermutationInvariantTraining')
    
    #def test_invalid_metric_reduction(self):
        """
        Test if the MetricFactory correctly loads a FeatureCoverageMetric class.
        """
        # Simulate a model object since FeatureCoverageMetric expects one
    #    model = Model(device=torch.device("cpu"))
    #    layer = SparseyLayer(
    #        autosize_grid=True,  
    #        grid_layout="rectangular",
    #        num_macs=1,
    #        num_cms_per_mac=1,
    #        num_neurons_per_cm=1,
    #        mac_grid_num_rows=1,
    #        mac_grid_num_cols=1,
    #        mac_receptive_field_size=3.0,
    #        prev_layer_num_cms_per_mac=1,
    #        prev_layer_num_neurons_per_cm=1,
    #        prev_layer_mac_grid_num_rows=2,
    #        prev_layer_mac_grid_num_cols=2,
    #        prev_layer_num_macs=4,
    #        prev_layer_grid_layout="rectangular",
    #        layer_index=0, 
    #        sigmoid_phi=5.0,
    #        sigmoid_lambda=28.0,
    #        saturation_threshold=0.5,
    #        permanence_steps=0.1, 
    #        permanence_convexity=0.1,
    #        activation_threshold_min=0.2, 
    #        activation_threshold_max=1.0, 
    #        min_familiarity=0.2, 
    #        sigmoid_chi=1.5 ,
    #        device=torch.device("cpu")
    #    )
    
    #    model.add_layer(layer)
        
    #    metric_class = MetricFactory.get_metric_class('feature_coverage')
    #    metric = MetricFactory.create_metric(
    #        'feature_coverage',
    #        params={},
    #        device=torch.device('cpu'),
    #        model=model,
    #        reduction='average_metric',
    #        comparison="max_by_layerwise_mean")
        # Create mock inputs, should use better tensors
    #    mock_input = torch.rand(10, 10)
    #    mock_labels = torch.randint(0, 5, (10,))  

        # Call compute to simulate metric computation
    #    result = metric.compute(model, mock_input, mock_labels, training=True)
    #    assert result is None
