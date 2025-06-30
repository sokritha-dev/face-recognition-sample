# utils/debouncer.py
from datetime import datetime, timedelta


class LoggerDebouncer:
    def __init__(self, interval_seconds: float):
        self.last_logged = {}  # {name: datetime}
        self.log_interval = timedelta(seconds=interval_seconds)

    def should_log(self, name: str, score: float, threshold: float) -> bool:
        """Log only if the interval has passed."""
        now = datetime.now()
        if (
            name not in self.last_logged
            or now - self.last_logged[name] > self.log_interval
        ):
            self.last_logged[name] = now
            return True

        return False
