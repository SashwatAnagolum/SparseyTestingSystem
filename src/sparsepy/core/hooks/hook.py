# -*- coding: utf-8 -*-

"""
Hook: file hlding the abstract Hook class represeting
    a PyTorch hook.
"""


import abc
from typing import Any

import torch


class Hook:
    """
    Hook: abstract base class for all Hooks in the system.
    """
    def __init__(self, module: torch.nn.Module) -> None:
        """
        Initializes the hook, and registers it with the model.

        Args:
            module (torch.nn.MOdule): model to be hooked into.
        """
        self.module = module
        self.hook_handles = []

        self.hook()


    @abc.abstractmethod
    def hook(self) -> None:
        """
        Register this hook with the model pased in during initialization.

        Concrete hooks need to implement this method to register
        the required hooks.
        """

    def remove(self) -> None:
        """
        Remove the hooks set by the class.
        """
        for handle in self.hook_handles:
            handle.remove()