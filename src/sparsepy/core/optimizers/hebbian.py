# -*- coding: utf-8 -*-

"""
Hebbian: file holding the Hebbian optimizer class.
"""


from typing import Iterator, Callable, Optional

import torch


class HebbianOptimizer(torch.optim.Optimizer):
    def __init__(self, params: Iterator[torch.nn.Parameter]):
        super().__init__(params, dict())

    def step(self, closure: Callable[[], torch.nn.Module]) -> None:
        """
        Performs a weight update.

        Args:
            closure: callable returning the model activations,
                which are used to compute pre-post correlations
                to decide which weights to increase.
        """
        activations = closure()

        for param in self.param_groups[0]['params']:
            param += torch.clamp(torch.randint(-10, 2, param.shape), min=0)
            torch.clamp(param, min=0, max=1, out=param)

        # print('Generated codes:\n-----------------------')

        # for layer in range(1, len(activations)):
        #     for mac_index in range(activations[layer].shape[1]):
        #         for wta_index in range(activations[layer].shape[2]):
        #             print(f'WTA module {wta_index} | MAC {mac_index} | Layer {layer}:')
        #             print(activations[layer][0][mac_index][wta_index].numpy())
        #             print()

        return