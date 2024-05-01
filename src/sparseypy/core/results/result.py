from datetime import datetime

class Result:
    """
    Result: class to store the results of a single step.
    Attributes:
        id (str): The id of the result.
        start_time (datetime): The start time of the result.
        end_time (datetime): The end time of the result.
    """
    def __init__(self):
        """
        Initializes the Result.
        """
        self.id = None
        self.start_time = datetime.now()
        self.end_time = None

    def mark_finished(self):
        """
        Mark the result as finished.
        """
        self.end_time = datetime.now()
