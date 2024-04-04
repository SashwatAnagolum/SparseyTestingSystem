# -*- coding: utf-8 -*-

"""
Transform Factory: class to build transforms.
"""

import inspect

import torch
from torchvision.transforms import v2

from sparseypy.core import transforms


class TransformFactory:
    """
    TransformFactory: factory class for constructing built-in system and
    PyTorch transforms.

    Attributes:
        allowed_modules (list[string]): the names of the transform classes shipped as
        part of the system that are allowed to be constructed with get_transform_class().
    """
    allowed_modules = {i for i in dir(transforms) if i[:2] != '__'} # set comprehension

    @staticmethod
    def get_transform_name(transform_name: str) -> str:
        """
        Converts a transform name from config file format ("transform_name") to
        transform class name format ("TransformName").

        Args:
            transform_name (str): the config-style transform name

        Returns:
            str: the class-style transform name
        """
        class_name = ''.join(
            [l[:1].upper() + l[1:] for l in transform_name.split('_')]
        )

        return class_name

    @staticmethod
    def get_transform_class(class_name: str):
        """
        Gets the class corresponding to the name passed in.
        Throws an error if the name is not valid.

        Args:
            class_name (str): the transform class name to create
        """
        # input is TransformName but standard for our classes is TransformNameTransform
        if class_name + 'Transform' in TransformFactory.allowed_modules:
            return getattr(transforms, class_name + 'Transform')
        # torchvision modules contain multiple classes; use reflection to get all the
        # torchvision class members' names and see if ours is among them
        elif class_name in [
            cls[0] for cls in inspect.getmembers(v2, inspect.isclass)
        ]:
            return getattr(v2, class_name)
        else:
            raise ValueError(f'Invalid transform name: {class_name}!')


    @staticmethod
    def create_transform(transform_name, **kwargs) -> v2.Transform:
        """
        Creates a transform passed in based on the transform name and kwargs.

        Args:
            transform_name (str): the name of the transform to create.
            **kwargs: arbitrary keyword arguments, passed to the transform
            class constructor.
        """
        transform_name = TransformFactory.get_transform_name(transform_name)

        transform_class = TransformFactory.get_transform_class(transform_name)

        if (transform_name in dir(v2)) and ('dtype' in kwargs):
            kwargs['dtype'] = getattr(torch, kwargs['dtype'])

        transform_obj = transform_class(**kwargs)

        return transform_obj
