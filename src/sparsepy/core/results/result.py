from datetime import datetime

class Result:
    def __init__(self):
        self.id = None
        self.start_time = datetime.now()
        self.end_time = None

    def mark_finished(self):
        self.end_time = datetime.now()
