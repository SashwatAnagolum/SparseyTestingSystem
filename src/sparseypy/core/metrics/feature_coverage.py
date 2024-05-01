import abc
import numpy as np
import torch

from typing import Optional, Callable

from sparseypy.access_objects.models.model import Model
from sparseypy.core.hooks import LayerIOHook
from sparseypy.core.metrics.metrics import Metric
from sparseypy.core.metrics.comparisons import max_by_layerwise_mean


class FeatureCoverageMetric(Metric):
    """
    FeatureCoverageMetric: metric computing the feature
        coverage of MACs and layers in a Sparsey model.

    Attributes:
        reduction (str): the type of reduction to apply
            onto the raw per-layer, per-sample feature coverage
            results.
        hook (LayerIOHook): the hook registered with the model
            being evaluated to obtain references to each layer,
            and layerwise inputs and outputs.
    """
    def __init__(self, model: torch.nn.Module,
                 device: torch.device,
                 reduction: Optional[str] = None,
                 best_value: Optional[Callable] = max_by_layerwise_mean) -> None:
        """
        Initializes the FeatureCoverageMetric object. 

        Args:
            model (torch.nn.Module): the model to compute feature
                coverage using.
            reduction (Optional[str]): the type of reduction
                to apply before returning the metric value.
        """
        super().__init__(
            model, "feature_coverage",
            best_value, device, reduction
        )

        self.hook = LayerIOHook(self.model)


    def _compute(self, m: Model, last_batch: torch.Tensor,
                labels: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Computes the feature coverage of a model for a given batch of inputs.

        Args:
            m (Model): Model to evaluate.
            last_batch (torch.Tensor): the model input for the current step
            labels (torch.Tensor): the model output for the current step
            training (bool): whether the model is training or evaluating

        Output:
            (float): feature coverage as a fraction.
        """
        layers, _, outputs = self.hook.get_layer_io()
        batch_size = last_batch.shape[0]
        last_batch = last_batch.view(batch_size, -1)

        feature_coverage = [
            torch.zeros(
                (batch_size, output.shape[1] + 1, last_batch.shape[1] + 1),
                dtype=torch.float32, device=self.device
            ) for output in outputs
        ]

        connections_view = layers[0].input_connections.view(
                1, *layers[0].input_connections.shape
        ).expand(batch_size, *layers[0].input_connections.shape)

        ones = torch.ones(
            (1, 1, 1), dtype=torch.float32, device=self.device
        ).expand(
            batch_size, outputs[0].shape[1],
            layers[0].input_connections.shape[-1]
        )

        feature_coverage[0].scatter_(2, connections_view, ones)

        for i in range(1, len(layers)):
            feature_coverage[i][:, :-1] = torch.mul(
                feature_coverage[i - 1][
                    :, layers[i].input_connections
                ],
                layers[i].is_active.view(*layers[i].is_active.shape, 1, 1)
            ).sum(2)

        feature_coverage_raw = [f[:, :-1, :-1] for f in feature_coverage]
        feature_coverage_percent = [
            torch.div(
                torch.logical_and(
                    f,
                    last_batch.view(
                        batch_size, 1,
                        *last_batch.shape[1:]
                    )
                ).sum(2),
                torch.sum(last_batch, 1, keepdim=True)
            )
            for f in feature_coverage_raw
        ]

        return torch.nested.nested_tensor(
            feature_coverage_percent, dtype=torch.float32,
            device=self.device
        )
