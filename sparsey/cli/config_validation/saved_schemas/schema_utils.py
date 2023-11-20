# -*- coding: utf-8 -*-

"""
Schema Utils: utility and helper functions for constructing schemas.
"""


from typing import Union


def is_positive(x: Union[int, float]) -> bool:
    """
    Returns whether a number is positive or not.

    Args:
        x: a float or int representing a number.

    Returns:
        a bool indicating whether x is positive or not.
    """
    return x > 0


def is_expected_len(x: list, expected_len: int) -> bool:
    """
    Returns whether a list is of the expected length or not.

    Args:
        x: a list.
        expected_len: an int representing the expected length
            of the list.

    Returns:
        a bool indicating whether x is the expected length
            or not.
    """    
    return len(x) == expected_len
