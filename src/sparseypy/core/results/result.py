from datetime import datetime
from typing import Optional, Union


class Result:
    """
    Result: class to store the results of a single step.
    Attributes:
        id (str): The id of the result.
        start_time (datetime): The start time of the result.
        end_time (datetime): The end time of the result.
    """
    def __init__(self, configs: Optional[dict] = None) -> None:
        """
        Initializes the Result.

        Args:
            configs (dict): the configuration for the process
                generating the Result (hpo / training).
        """
        self.id = None
        self.start_time = datetime.now()
        self.end_time = None
        self.configs = configs

        if self.configs is None:
            self.configs = dict()


    def mark_finished(self):
        """
        Mark the result as finished.
        """
        self.end_time = datetime.now()


    def get_config(self, config_name: str) -> Union[any, None]:
        """
        Returns the configuration associated with the
        config_name passed in. Returns None if
        config_name does not exist as a key in self.configs.

        Args:
            config_name (str): the name of the config to get.

        Returns:
            (Union[any, None]): the config value, if it exists.
        """
        return self.configs.get(config_name, None)


    def get_configs(self) -> dict:
        """
        Get the configurations for the training run.
        Returns:
            (dict): The configurations for the training run.
        """
        return self.configs


    def add_config(self, name, config) -> None:
        """
        Add a configuration to the result.

        Args:
            name (str): The name of the configuration.
            config: The configuration to add.
        """
        self.configs[name] = config
