# -*- coding: utf-8 -*-

"""
Transform Factory: class to build transforms.
"""

import inspect

import torch

from torchvision.transforms import v2

from sparseypy.core import transforms


class TransformFactory:
    allowed_modules = set([i for i in dir(transforms) if i[:2] != '__'])

    @staticmethod
    def get_transform_name(transform_name):
        class_name = ''.join(
            [l.capitalize() for l in transform_name.split('_')]
        )

        return class_name

    @staticmethod
    def get_transform_class(class_name):
        """
        Gets the class corresponding to the name passed in.
        Throws an error if the name is not valid.
        """
        # input is TransformName but standard for our classes is TransformNameTransform
        if class_name + 'Transform' in TransformFactory.allowed_modules:
            return getattr(transforms, class_name + 'Transform')
        # torchvision modules contain multiple classes; use reflection to get all the
        # torchvision class members' names and see if ours is among them
        elif class_name in [cls[0] for cls in inspect.getmembers(v2, inspect.isclass)]:
            return getattr(v2, class_name)
        else:
            raise ValueError(f'Invalid transform name: {class_name}!')
    

    @staticmethod
    def create_transform(transform_name, **kwargs) -> v2.Transform:
        """
        Creates a transform passed in based on the transform name and kwargs.
        """
        transform_name = TransformFactory.get_transform_name(transform_name)

        transform_class = TransformFactory.get_transform_class(transform_name)

        if (transform_name in dir(v2)) and ('dtype' in kwargs):
            kwargs['dtype'] = getattr(torch, kwargs['dtype'])

        transform_obj = transform_class(**kwargs)

        return transform_obj
