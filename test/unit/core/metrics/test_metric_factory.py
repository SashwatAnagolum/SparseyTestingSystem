# -*- coding: utf-8 -*-

"""
Test Layer Factory: test cases for the LayerFactory class.
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
        """
        metric = MetricFactory.get_metric_class('match_accuracy')

        print(metric)

        assert issubclass(metric, Metric)

    def test_valid_metric_feature_coverage(self):
        """
        Test if the MetricFactory correctly loads a FeatureCoverageMetric class.
        """
        # Simulate a model object since FeatureCoverageMetric expects one
        model = Model()
        layer = SparseyLayer(
            autosize_grid=True,  
            grid_layout="rectangular",
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
    
        model.add_layer(layer)
        # Attempt to get the FeatureCoverageMetric class from the factory
        metric_class = MetricFactory.get_metric_class('feature_coverage')

        # Verify that the returned class is FeatureCoverageMetric
        assert metric_class == FeatureCoverageMetric, "MetricFactory did not return the expected metric class."

        # Instantiate the metric to verify it can be created successfully
        metric = metric_class(model=model, reduction='mean')

        # Create mock inputs, should use better tensors
        mock_input = torch.rand(10, 10)
        mock_labels = torch.randint(0, 5, (10,))  

        # Call compute to simulate metric computation
        result = metric.compute(model, mock_input, mock_labels, training=True)

        # Basic check to verify that result is a tensor
        assert isinstance(result, torch.Tensor), "The compute method did not return a tensor."


    def test_invalid_metric_name(self) -> None:
        """
        Tests whether the LayerFactory throws an error  
        if we provide it with a invalid layer name.
        """
        with pytest.raises(ValueError):
            MetricFactory.get_metric_class('martian_accuracy')
    
    def test_invalid_metric_reduction(self):
        """
        Test if the MetricFactory correctly loads a FeatureCoverageMetric class.
        """
        # Simulate a model object since FeatureCoverageMetric expects one
        model = Model()
        layer = SparseyLayer(
            autosize_grid=True,  
            grid_layout="rectangular",
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
    
        model.add_layer(layer)
        
        metric_class = MetricFactory.get_metric_class('feature_coverage')
        metric = metric_class(model=model, reduction='average_metric')
        # Create mock inputs, should use better tensors
        mock_input = torch.rand(10, 10)
        mock_labels = torch.randint(0, 5, (10,))  

        # Call compute to simulate metric computation
        result = metric.compute(model, mock_input, mock_labels, training=True)
        assert result is None

"""def test_sparsey_layer(self) -> None:
"""
#Tests whether the LayerFactory correctly constructs a Sparsey layer
#or not.
"""
        metric_obj = MetricFactory.create_metric(
            'basis_set_size',
            save=True
        )

        #data = torch.randint(
        #    0, 2, (4, 9, 10, 10), dtype=torch.float32
        #)

        #assert metric_obj(data).shape == (4, 10, 8, 4)
"""