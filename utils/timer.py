import time


class Timer:
    def __init__(self, name=None, auto_log=True):
        self.name = name
        self.auto_log = auto_log
        self.start_time = None
        self.end_time = None
        self.duration = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self  # you can use `as t` to access .duration

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        if self.auto_log:
            label = f"[{self.name}]" if self.name else "[Timer]"
            print(f"{label} took {self.duration:.4f}s")
