# -*- coding: utf-8 -*-

"""
Layer IO: file hlding the LayerIOHook class.
"""


import torch

from itertools import chain

from sparsepy.core.hooks.hook import Hook
from sparsepy.core.model_layers.sparsey_layer import SparseyLayer


class LayerIOHook(Hook):
    """
    Layer IO Hook: simple hook to get the output
        and input of a layer.
    """
    def __init__(self, module: torch.nn.Module, flatten = True) -> None:
        """
        Initializes the hook, and registers it with the model.

        Args:
            module (torch.nn.MOdule): model to be hooked into.
        """
        super().__init__(module)

        # runs before forward is invoked (only on top-level Model)
        # on pre hook, reinitialize the lists
        self.input_list = []
        self.output_list = []
        self.layer_list = []

        self.flatten = flatten

        for m in module.modules():
            if isinstance(m, SparseyLayer):
                # WARNING this approach will not work correctly with nested SparseyLayers
                self.input_list.append([])
                self.output_list.append([])
                self.layer_list.append([])

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
        self.input_list = []
        self.output_list = []
        self.layer_list = []

        # creates implicit requirement that the module passed to the pre hook is the top-level module
        for m in module.modules():
            if isinstance(m, SparseyLayer):
                # WARNING this approach will not work correctly with nested SparseyLayers
                self.input_list.append([])
                self.output_list.append([])
                self.layer_list.append([])


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
        self.layer_list[module.layer_index].append(module)
        self.output_list[module.layer_index].append(output)
        self.input_list[module.layer_index].append(input[0])


    def get_layer_io(self):
        if self.flatten:
            return list(chain.from_iterable(self.layer_list)), list(chain.from_iterable(self.input_list)), list(chain.from_iterable(self.output_list))
        else:
            return self.layer_list, self.input_list, self.output_list