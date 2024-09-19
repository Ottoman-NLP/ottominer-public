import sys
import time

class ProgressBar:
    def __init__(self, total, prefix='', suffix='', decimals=1, length=50, fill='█', print_end="\r"):
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.length = length
        self.fill = fill
        self.print_end = print_end
        self.start_time = time.time()
        self.iteration = 0

    def print(self, iteration):
        self.iteration = iteration
        percent = ("{0:." + str(self.decimals) + "f}").format(100 * (iteration / float(self.total)))
        filled_length = int(self.length * iteration // self.total)
        bar = self.fill * filled_length + '-' * (self.length - filled_length)
        elapsed_time = time.time() - self.start_time
        estimated_total_time = elapsed_time * (self.total / iteration) if iteration > 0 else 0
        estimated_remaining_time = estimated_total_time - elapsed_time
        time_suffix = f'| {self.format_time(elapsed_time)}<{self.format_time(estimated_remaining_time)}'
        print(f'\r{self.prefix} |{bar}| {percent}% {self.suffix} {time_suffix}', end=self.print_end)
        if iteration == self.total:
            print()

    @staticmethod
    def format_time(seconds):
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f'{h:02d}:{m:02d}:{s:02d}'

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.iteration != self.total:
            self.print(self.total)