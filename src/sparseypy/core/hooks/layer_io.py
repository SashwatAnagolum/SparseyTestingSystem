# -*- coding: utf-8 -*-

"""
Layer IO: file hlding the LayerIOHook class.
"""


import torch

from sparseypy.core.hooks.hook import Hook


class LayerIOHook(Hook):
    """
    Layer IO Hook: simple hook to get the output
        and input of a Sparsey model.
    """
    def __init__(self, module: torch.nn.Module, flatten = False) -> None:
        """
        Initializes the hook, and registers it with the Sparsey model.

        Args:
            module (torch.nn.MOdule): model to be hooked into.
            flatten (boolean): whether to flatten the model structure
            into a 1d list for return. Default false.
            (deprecated, will be removed soon)
        """
        super().__init__(module)

        # save number of layers at creation so we don't have to recompute it
        self.input_list = []
        self.output_list = []
        self.layer_list = []

        self.flatten = flatten


    def hook(self) -> None:
        """
        Register this hook with the model pased in during initialization.

        Concrete hooks need to implement this method to register
        the required hooks.
        """
        # get all the layers in the network
        for module in self.module.children():
            handle = module.register_forward_hook(self.forward_hook)
            self.hook_handles.append(handle)

        # pre hook is attached only to the top-level module
        self.module.register_forward_pre_hook(self.pre_hook)


    def pre_hook(self, module: torch.nn.Module, input: torch.Tensor) -> None:
        """
        Pre-hook to capture the input of the model.
        Args:
            module (torch.nn.Module): the module that the hook was
                registered to.
            input (torch.Tensor): module input
        """
        self.input_list = []
        self.output_list = []
        self.layer_list = []


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
        # creates dependency on layer_index inside the MAC this is attached to; needs reevaluating if we support nested blocks
        self.layer_list.append(module)
        self.output_list.append(output)
        self.input_list.append(input[0])


    def get_layer_io(self) -> tuple[
        list[torch.nn.Module], list[torch.Tensor],
        list[torch.Tensor]]:
        """
        Returns the captured layers, inputs, and outputs.

        Returns:
            (tuple[list[torch.nn.Module], list[torch.Tensor],
                list[torch.Tensor]]
            ): a tuple containing the layers, input, and outputs
            of the model.
        """
        return self.layer_list, self.input_list, self.output_list
