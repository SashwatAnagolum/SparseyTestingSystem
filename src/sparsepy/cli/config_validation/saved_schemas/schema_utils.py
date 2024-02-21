# -*- coding: utf-8 -*-

"""
Schema Utils: utility and helper functions for constructing schemas.
"""


from typing import Union, Callable


def is_positive(x: Union[int, float]) -> bool:
    """
    Returns whether a number is positive or not.

    Args:
        x: a float or int representing a number.

    Returns:
        a bool indicating whether x is positive or not.
    """
    return x > 0


def is_nonnegative(x: Union[int, float]) -> bool:
    """
    Returns whether a number is nonnegative or not.

    Args:
        x: a float or int representing a number.

    Returns:
        a bool indicating whether x is nonnegative or not.
    """
    return x >= 0


def all_elements_satisfy(x: list, cond: Callable):
    """
    Returns whether all elements in a list satisfy a condition or not.

    Args:
        x (list): the list to check.

    Returns:
        (bool): whether all elements satisfy the condtion or not.
    """
    for element in x:
        if not cond(element):
            return False

    return True


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


def all_elements_are_same_type(x: list):
    """
    Returns whether all elements in a list are the same type or not.

    Args:
        x (list): the list to be checked.

    Returns:
        (bool): whether all elements are the same type or not.
    """
    list_type = type(x[0])

    for element in x:
        if not isinstance(element, list_type):
            return False

    return True
