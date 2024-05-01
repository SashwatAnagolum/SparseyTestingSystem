# -*- coding: utf-8 -*-

"""
Optimizer Factory: file holding the Optimizer Factory class.
"""


import torch

from sparseypy.core import optimizers


class OptimizerFactory:
    """
    Factory class for constructing Optimizers.

    Contains methods for checking for the existence of optimizers by name and instantiating
    optimizer instances using reflection.

    Attributes:
        allowed_modules (set[str]): the valid optimizers available for use in the system
    """
    allowed_modules = set([i for i in dir(optimizers) if i[:2] != '__'])

    @staticmethod
    def get_optimizer_class(opt_name):
        """
        Gets the class corresponding to the name passed in.
        Throws an error if the name is not valid.
        """
        class_name = ''.join(
            [l.capitalize() for l in opt_name.split('_')] + ['Optimizer']
        )

        if class_name in OptimizerFactory.allowed_modules:
            return getattr(optimizers, class_name)
        elif opt_name in dir(torch.optim):
            return getattr(torch.optim, opt_name)
        else:
            raise ValueError('Invalid optimizer name!')


    @staticmethod
    def create_optimizer(opt_name, **kwargs) -> torch.optim.Optimizer:
        """
        Creates a layer passed in based on the layer name and kwargs.
        """
        opt_class = OptimizerFactory.get_optimizer_class(opt_name)

        # Before instantiation, check if 'thresh' is required and present
        #if opt_name.lower() == 'hebbian' and 'thresh' not in kwargs:
            #raise ValueError("Missing required 'thresh' parameter for HebbianOptimizer")

        opt_obj = opt_class(**kwargs)

        return opt_obj
