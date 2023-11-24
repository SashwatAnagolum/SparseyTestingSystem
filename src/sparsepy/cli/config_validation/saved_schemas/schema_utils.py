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


def is_between(x: Union[int, float],
               range_start: Union[int, float],
               range_end: Union[int, float]) -> bool:
    """
    Returns whether a number is within a given range or not.

    Args:
        x: a float or int representing a number.
        range_start: a float or int representing the start of the range
            (inclusive).
        range_end: a float or int representing the end of the range
            (inclusive).

    Returns:
        a bool indicating whether x is in the given range or not.
    """
    return (x >= range_start) and (x <= range_end)
