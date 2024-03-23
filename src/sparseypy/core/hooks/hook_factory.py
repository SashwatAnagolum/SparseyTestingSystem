# -*- coding: utf-8 -*-

"""
Hook Factory: factory for creating model hooks.
"""


import torch

from sparseypy.core import hooks
from sparseypy.core.hooks.hook import Hook


class HookFactory:
    """
    Hook Factory: Factory for generating hooks.
    """
    @staticmethod
    def create_hook(hook_name: str, model: torch.nn.Module) -> Hook:
        """
        Creates a new hook based on the name passed in, and initializes
        the hook with the model passed in. 

        Args:
            hook_name (str): name of the hook to create
            model (torch.nn.Module): model to initialize the hook with.
        """
        hook_name = ''.join(
            [i.capitalize() for i in hook_name.split('_')] + ['Hook']
        )

        if hook_name not in dir(hooks):
            raise ValueError(f'Invalid hook name: {hook_name}!')

        hook_obj = getattr(hooks, hook_name)(model)

        return hook_obj
