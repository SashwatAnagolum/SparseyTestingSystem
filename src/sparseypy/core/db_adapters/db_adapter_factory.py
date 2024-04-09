"""
db_adapter_factory.py - file containing the DbAdapterFactory class
"""

import inspect

from sparseypy.core import db_adapters
from sparseypy.core.db_adapters.db_adapter import DbAdapter


class DbAdapterFactory:
    """
    Factory class for database adapters.

    Attributes:
        allowed_modules (set): the valid database adapter classes this factory can instantiate.
    """
    allowed_modules = {i[0] for i in inspect.getmembers(db_adapters, inspect.isclass) if i[:2] != '__'}

    @staticmethod
    def get_db_adapter_class(db_adapter_name: str):
        """
        Gets the class corresponding to the name passed in.
        Throws an error if the name is not valid.
        """
        class_name = ''.join(
            [l.capitalize() for l in db_adapter_name.split('_')] + ['DbAdapter']
        )

        #if class_name in DbAdapterFactory.allowed_modules:
        return getattr(db_adapters, class_name)
        # not implemented yet - wrapping PyTorch metrics requires additional consideration
        # (and finding a way in PyTorch to determine what functions count as metrics)
        #elif metric_name in dir(torch.optim):
        #    return getattr(torch.optim, opt_name)
        #else:
        #    raise ValueError('Invalid DbAdapter name!')

    @staticmethod
    def create_db_adapter(db_adapter_name, **kwargs) -> DbAdapter:
        """
        Creates a database adapter based on the passed-in name and kwargs.

        Args:
            db_adapter_name (str): the name of the database adapter class to create.
            **kwargs: arbitrary keyword arguments passed to the class constructor.
        """
        db_adapter_class = DbAdapterFactory.get_db_adapter_class(db_adapter_name)

        db_adapter_obj = db_adapter_class(**kwargs)

        return db_adapter_obj

    @staticmethod
    def is_valid_db_adapter_class(db_adapter_name: str) -> bool:
        """
        Checks whether a database adapter class exists corresponding to the passed-in name.

        Args:
            db_adapter_name: the name of the database adapter class to check for.
        """
        class_name = ''.join(
            [l.capitalize() for l in db_adapter_name.split('_')] + ['DbAdapter']
        )

        return class_name in DbAdapterFactory.allowed_modules
