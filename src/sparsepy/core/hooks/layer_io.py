# -*- coding: utf-8 -*-

"""
Layer IO: file hlding the LayerIOHook class.
"""


import torch

from sparsepy.core.hooks.hook import Hook


class LayerIOHook(Hook):
    """
    Layer IO Hook: simple hook to get the output
        and input of a layer.
    """
    def __init__(self, module: torch.nn.Module) -> None:
        """
        Initializes the hook, and registers it with the model.

        Args:
            module (torch.nn.MOdule): model to be hooked into.
        """
        super().__init__(module)

        self.input_list = []
        self.output_list = []
        self.layer_list = []

    def hook(self) -> None:
        """
        Register this hook with the model pased in during initialization.

        Concrete hooks need to implement this method to register
        the required hooks.
        """
        # get all the submodules in the network
        for module in self.module.modules():
            # if that module has no children (= it is a MAC)
            if next(module.children(), None) is None:
                # then add it to the hook handles
                handle = module.register_forward_hook(self.forward_hook)
                self.hook_handles.append(handle)

        self.module.register_forward_pre_hook(self.pre_hook)


    def pre_hook(self, module: torch.nn.Module, input: torch.Tensor) -> None:
        self.layer_list = []
        self.input_list = []
        self.output_list = []


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
        self.layer_list.append(module)
        self.output_list.append(output)
        self.input_list.append(input[0])


    def get_layer_io(self):
        return self.layer_list, self.input_list, self.output_list