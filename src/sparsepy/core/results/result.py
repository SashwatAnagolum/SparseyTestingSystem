from datetime import datetime

class Result:
    def __init__(self, id: str, start_time: datetime, end_time: datetime):
        self.id = id
        self.start_time = start_time
        self.end_time = end_time
