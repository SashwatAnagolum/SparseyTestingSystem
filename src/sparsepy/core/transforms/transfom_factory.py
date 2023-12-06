# -*- coding: utf-8 -*-

"""
Transform Factory: class to build transforms.
"""


from torchvision.transforms import v2

from sparsepy.core import transforms


class TransformFactory:
    allowed_modules = set([i for i in dir(transforms) if i[:2] != '__'])

    @staticmethod
    def get_transform_class(transform_name):
        """
        Gets the class corresponding to the name passed in.
        Throws an error if the name is not valid.
        """
        class_name = ''.join(
            [l.capitalize() for l in transform_name.split('_')]
        )

        if class_name in TransformFactory.allowed_modules:
            return getattr(transforms, class_name)
        elif class_name in dir(v2):
            return getattr(v2, class_name)
        else:
            raise ValueError(f'Invalid transform name: {class_name}!')
    

    @staticmethod
    def create_transform(layer_name, **kwargs) -> v2.Transform:
        """
        Creates a transform passed in based on the transform name and kwargs.
        """
        transform_class = TransformFactory.get_transform_class(layer_name)

        transform_obj = transform_class(**kwargs)

        return transform_obj
