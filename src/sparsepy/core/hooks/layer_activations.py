# -*- coding: utf-8 -*-

"""
Layer Activations: file hlding the LayerActivationsHook class.
"""


import torch

from sparsepy.core.hooks.hook import Hook


class LayerActivationsHook(Hook):
    """
    Layer Activations Hook: simple hook to get the output
        of a layer.
    """
    def __init__(self, module: torch.nn.Module) -> None:
        """
        Initializes the hook, and registers it with the model.

        Args:
            module (torch.nn.MOdule): model to be hooked into.
        """
        super().__init__(module)

        self.layer_output = []


    def hook(self) -> None:
        """
        Register this hook with the model pased in during initialization.

        Concrete hooks need to implement this method to register
        the required hooks.
        """
        for module in self.module.modules():
            handle = module.register_forward_hook(self.forward_hook)
            self.hook_handles.append(handle)

        module.register_forward_pre_hook(self.pre_hook)

    def pre_hook(self, module: torch.nn.Module, input: torch.Tensor) -> None:
        self.layer_output = []


    def forward_hook(self, module: torch.nn.Module,
                 input: torch.Tensor, output: torch.Tensor) -> None:
        """
        Call the hook.

        Args:
            module (torch.nn.Module): the module that the hook was
                registered to.
            input (torch.Tensor): module input
            output (torch.Tensor): module output
        """
        self.layer_output.append(output.clone().detach())

        print(self.layer_output[-1].shape)
