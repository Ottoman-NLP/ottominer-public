import itertools
import sys
import time
import threading
from sys_metrics import SysMetrics

class AnimatedProgress:
    def __init__(self, directory):
        self.directory = directory
        self.done = False
        self.current_task = "Processing"
        self.next_target = "Unknown"
        self.progress = 0
        self.total = 100
        self.sys_metrics = SysMetrics()
        self.metrics_thread = threading.Thread(target=self.collect_metrics)
        self.metrics_thread.start()

    def collect_metrics(self):
        while not self.done:
            self.sys_metrics.get_system_metrics(self.directory)
            time.sleep(5)  # Adjust the interval as needed

    def animate(self):
        for c in itertools.cycle(['|', '/', '-', '\\']):
            if self.done:
                break
            sys.stdout.write(f'\r{self.current_task} {c} {self.progress}% - Next: {self.next_target} - CPU: {self.sys_metrics.metrics["cpu_usage"]}% GPU: {self.sys_metrics.metrics["gpu_usage"]}%')
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write('\rDone!     \n')

    def update_progress(self, progress, task, next_target):
        self.progress = progress
        self.current_task = task
        self.next_target = next_target

    def finish(self):
        self.done = True
        self.metrics_thread.join()

if __name__ == "__main__":
    directory = "/path/to/working/directory"  # Example path, adjust according to use
    progress = AnimatedProgress(directory)
    try:
        for i in range(100):
            progress.update_progress(i + 1, "Processing", "Next Phase")
            time.sleep(0.1)  # Simulate task duration
    finally:
        progress.finish()

