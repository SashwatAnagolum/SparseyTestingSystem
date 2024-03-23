# -*- coding: utf-8 -*-

"""
Layer IO: file hlding the LayerIOHook class.
"""


import torch

from itertools import chain

from sparseypy.core.hooks.hook import Hook
from sparseypy.core.model_layers.sparsey_layer import SparseyLayer


class LayerIOHook(Hook):
    """
    Layer IO Hook: simple hook to get the output
        and input of a layer.
    """
    def __init__(self, module: torch.nn.Module, flatten = False) -> None:
        """
        Initializes the hook, and registers it with the model.

        Args:
            module (torch.nn.MOdule): model to be hooked into.
            flatten (boolean): whether to flatten the model structure into a 1d list for return. Default false.
        """
        super().__init__(module)

        # WARNING creates some potentially undesirable assumptions about network structure
        # also this approach requires .modules() to return all MACs in order by index within layer and in order by layer
        # save number of layers at creation so we don't have to recompute it
        self.num_layers = sum([1 for m in module.modules() if isinstance(m, SparseyLayer)])
        # create empty 2D list with one inner list for each layer
        self.input_list = [[] for i in range(self.num_layers)]
        self.output_list = [[] for i in range(self.num_layers)]
        self.layer_list = [[] for i in range(self.num_layers)]

        self.flatten = flatten

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
        # pre hook is attached only to the top-level module
        self.module.register_forward_pre_hook(self.pre_hook)


    def pre_hook(self, module: torch.nn.Module, input: torch.Tensor) -> None:
        self.input_list = [[] for i in range(self.num_layers)]
        self.output_list = [[] for i in range(self.num_layers)]
        self.layer_list = [[] for i in range(self.num_layers)]


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
        self.layer_list[module.layer_index].append(module)
        self.output_list[module.layer_index].append(output)
        self.input_list[module.layer_index].append(input[0])


    def get_layer_io(self):
        if self.flatten:
            return list(chain.from_iterable(self.layer_list)), list(chain.from_iterable(self.input_list)), list(chain.from_iterable(self.output_list))
        else:
            return self.layer_list, self.input_list, self.output_list