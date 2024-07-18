# @author: <rekurrenz>
# @license: MIT
# @version: 1.0.0

#anim/itercycle.py -- a animation module for progress bars

from utils import get_system_metrics
import itertools
import sys
import time
import logging
import os

class AnimatedProgress:
    def __init__(self):
        self.done = False
        self.current_task = ""
        self.next_target = ""
        self.progress = 0
        self.total = 100 

        self.system_metrics = {
            'gpu': 0,  # GPU usage
            'cpu': 0,  # CPU usage
            'switches': 0,  # Context switches
            'fs_in': 0,  # File system read count
            'fs_out': 0,  # File system write count
            'size': 0,  # Folder/file size in MB
            'hash': 0,  # Hash of progress
            'eta': 0  # Estimated time of arrival
        }

        self.start_time = time.time()

    def animate_general(self):
        for c in itertools.cycle(['|', '/', '-', '\\']):
            if self.done:
                break
            sys.stdout.write(f'\r{self.current_task} {c} {self.progress}% - Next: {self.next_target}')
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write('\rDone!     \n')

    def get_folder_size(self, directory):
        total_size = 0

        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)

            def simple_hash(data):
                try:
                    import hashlib
                    return int(hashlib.md5(data.encode()).hexdigest(), 16)
                except Exception as e:
                    logging.error(e)
                    return 0

            self.system_metrics['hash'] = simple_hash(str(self.progress) + str(total_size))
            self.system_metrics['size'] = total_size / (1024 * 1024)  # Convert to MB

        except Exception as e:
            logging.error(e)
            self.system_metrics['hash'] = 0
            self.system_metrics['size'] = 0

    def get_sys_metrics(self):
        try:
            import cupy as cp
            import psutil
            self.system_metrics['gpu'] = cp.cuda.Device(0).use().memoryInfo().free / cp.cuda.Device(0).use().memoryInfo().total * 100
            self.system_metrics['cpu'] = psutil.cpu_percent()
            self.system_metrics['switches'] = psutil.cpu_stats().ctx_switches
            self.system_metrics['fs_in'] = psutil.disk_io_counters().read_count
            self.system_metrics['fs_out'] = psutil.disk_io_counters().write_count

            def simple_hash(data):
                try:
                    import hashlib
                    return int(hashlib.md5(data.encode()).hexdigest(), 16)
                except Exception as e:
                    logging.error(e)
                    return 0

            self.system_metrics['hash'] = simple_hash(str(self.progress) + str(self.total))
            self.system_metrics['eta'] = (self.total - self.progress) * (time.time() - self.start_time) / self.progress if self.progress > 0 else 0

        except Exception as e:
            logging.error(e)
            raise Exception("Error in get_sys_metrics")
        
        return self.system_metrics

    def animate_detailed(self):
        for c in itertools.cycle(['.', 'o', 'O', '0']):
            if self.done:
                break

            system_metrics = get_system_metrics(self.progress, self.total, self.start_time, self.directory)

            sys.stdout.write(f'\r{self.current_task} {c} {self.progress}% [{system_metrics}] - Next: {self.next_target}')
            sys.stdout.flush()


if __name__ == "__main__":
    progress = AnimatedProgress()
    progress.current_task = "Loading"
    progress.next_target = "Processing"
    progress.animate_general()
