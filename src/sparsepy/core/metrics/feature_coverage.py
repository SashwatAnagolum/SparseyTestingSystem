import abc
import numpy as np
import torch

from typing import Optional

from sparsepy.access_objects.models.model import Model
from sparsepy.core.hooks import LayerIOHook
from sparsepy.core.metrics.metrics import Metric
from sparsepy.core.metrics.comparisons import max_by_layerwise_mean

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
                 reduction: Optional[str] = None) -> None:
        """
        Initializes the FeatureCoverageMetric object. 

        Args:
            model (torch.nn.Module): the model to compute feature
                coverage using.
            reduction (Optional[str]): the type of reduction
                to apply before returning the metric value.
        """
        super().__init__(model, "feature_coverage", max_by_layerwise_mean)

        self.reduction = reduction
        self.hook = LayerIOHook(self.model)


    def compute(self, m: Model, last_batch: torch.Tensor,
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
        layers, _, _ = self.hook.get_layer_io()

        last_batch = last_batch.view(last_batch.shape[0], -1).bool()
        rf_cache = [[] for i in range(len(layers))]
        layer_masks = [
            torch.zeros(last_batch.shape, dtype=torch.bool)
            for i in range(len(layers))    
        ]

        for mac in layers[0]:
            rf_cache[0].append(
                torch.zeros(last_batch.shape, dtype=torch.bool)
            )

            rf_cache[0][-1][:, mac.input_filter] = True
            rf_cache[0][-1] = torch.logical_and(
                rf_cache[0][-1], mac.is_active.unsqueeze(1)
            )

            layer_masks[0] = torch.bitwise_or(
                layer_masks[0], rf_cache[0][-1]
            )

        for layer_index, layer in zip(range(1, len(layers)), layers[1:]):
            for mac in layer:
                rf_cache[layer_index].append(
                    torch.zeros(last_batch.shape, dtype=torch.bool)
                )

                for source_mac_index in mac.input_filter:
                    rf_cache[layer_index][-1] = torch.bitwise_or(
                        rf_cache[layer_index][-1],
                        rf_cache[layer_index - 1][source_mac_index]
                    )

                rf_cache[layer_index][-1] = torch.logical_and(
                    rf_cache[layer_index][-1],
                    mac.is_active.unsqueeze(1)
                )

                layer_masks[layer_index] = torch.bitwise_or(
                    layer_masks[layer_index],
                    rf_cache[layer_index][-1]
                )

        feature_coverage_values = []
        active_input_pixel_count = torch.count_nonzero(last_batch, 1)

        for layer_mask in layer_masks:
            feature_coverage_values.append(
                torch.nan_to_num(
                    torch.count_nonzero(
                        torch.bitwise_and(layer_mask, last_batch), 1
                    ) / active_input_pixel_count,
                    1.0
                )
            )

        feature_coverage_values = torch.stack(
            feature_coverage_values
        )

        if self.reduction is None or self.reduction == "none":
            return feature_coverage_values
        elif self.reduction == "sum":
            return torch.sum(torch.mean(feature_coverage_values, 1))
        elif self.reduction == "mean":
            return torch.mean(feature_coverage_values)
        else:
            return None
