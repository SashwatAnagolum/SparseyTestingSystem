# -*- coding: utf-8 -*-

"""
Hebbian: file holding the Hebbian optimizer class.
"""


from typing import Iterator, Callable, Optional

import torch


class Hebbian(torch.optim.Optimizer):
    def __init__(self, params: Iterator[torch.nn.Parameter]):
        super().__init__(params, dict())

    def step(self, closure: Callable[[], list[torch.Tensor]]) -> None:

        pass