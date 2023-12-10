# -*- coding: utf-8 -*-

"""
Test Layer Factory: test cases for the LayerFactory class.
"""


import pytest
import torch

from sparsepy.core import metrics
from sparsepy.core.metrics.metric_factory import MetricFactory
from sparsepy.core.metrics.metrics import Metric


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
        metric = MetricFactory.get_metric_class('exact_match_accuracy')

        print(metric)

        assert issubclass(metric, Metric)


    def test_invalid_metric_name(self) -> None:
        """
        Tests whether the LayerFactory throws an error  
        if we provide it with a invalid layer name.
        """
        with pytest.raises(ValueError):
            MetricFactory.get_metric_class('martian_accuracy')


    def test_sparsey_layer(self) -> None:
        """
        Tests whether the LayerFactory correctly constructs a Sparsey layer
        or not.
        """
        metric_obj = MetricFactory.create_metric(
            'basis_set_size',
            save=True
        )

        #data = torch.randint(
        #    0, 2, (4, 9, 10, 10), dtype=torch.float32
        #)

        #assert metric_obj(data).shape == (4, 10, 8, 4)
