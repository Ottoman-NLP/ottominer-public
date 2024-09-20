import sys
import time
from datetime import datetime

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
        self.update_interval = max(1, total // 100)  # Update every 1% progress
        self.last_update_time = self.start_time
        self.last_update_iteration = 0

    def print(self, iteration):
        self.iteration = iteration
        current_time = time.time()
        
        if iteration % self.update_interval == 0 or iteration == self.total:
            percent = ("{0:." + str(self.decimals) + "f}").format(100 * (iteration / float(self.total)))
            filled_length = int(self.length * iteration // self.total)
            bar = self.fill * filled_length + '-' * (self.length - filled_length)
            
            elapsed_time = current_time - self.start_time
            estimated_total_time = elapsed_time * (self.total / iteration) if iteration > 0 else 0
            estimated_remaining_time = max(0, estimated_total_time - elapsed_time)
            
            time_suffix = (f'| {self.format_time(elapsed_time)} < {self.format_time(estimated_remaining_time)} | '
                           f'ETA: {datetime.fromtimestamp(current_time + estimated_remaining_time).strftime("%Y-%m-%d %H:%M:%S")}')
            
            iterations_per_second = (iteration - self.last_update_iteration) / (current_time - self.last_update_time)
            speed_suffix = f'| Speed: {iterations_per_second:.2f} it/s'
            
            print(f'\r{self.prefix} |{bar}| {percent}% {self.suffix} {time_suffix} {speed_suffix}', end=self.print_end)
            
            if iteration == self.total:
                print()
            
            self.last_update_time = current_time
            self.last_update_iteration = iteration
        
        sys.stdout.flush()

    @staticmethod
    def format_time(seconds):
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        if d > 0:
            return f'{d}d {h:02d}:{m:02d}:{s:02d}'
        else:
            return f'{h:02d}:{m:02d}:{s:02d}'

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.iteration != self.total:
            self.print(self.total)