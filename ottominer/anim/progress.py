import itertools
import threading
import time
import sys

class ProgressBar:
    def __init__(self):
        self.done = False
        self.current_task = ""
        self.next_target = ""
        self.progress = 0
        self.progress_details = {}
        self.start_time = time.time()
        self.directory = ""

    def animate_general(self):
        for c in itertools.cycle(['|', '/', '-', '\\']):
            if self.done:
                break
            sys.stdout.write(f'\r{self.current_task} {c} {self.progress}% - Next: {self.next_target}')
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write('\rDone!     \n')

    def animate_detailed(self):
        for c in itertools.cycle(['.', 'o', 'O', '0']):
            if self.done:
                break
            details = (
                f"GPU: {self.progress_details.get('gpu', 0)}%, "
                f"CPU: {self.progress_details.get('cpu', 0)}%, "
                f"Sw: {self.progress_details.get('switches', 0)}, "
                f"FS In: {self.progress_details.get('fs_in', 0)}, "
                f"FS Out: {self.progress_details.get('fs_out', 0)}, "
                f"Size: {self.progress_details.get('size', 0)} MB, "
                f"Hash: {self.progress_details.get('hash', 0)} H/s, "
                f"ETA: {self.progress_details.get('eta', 0)} s"
            )
            sys.stdout.write(f'\r{self.current_task} {c} {self.progress}% [{details}] - Next: {self.next_target}')
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write('\rDone!     \n')

    def start(self, task, next_target, progress_type="general", directory=""):
        self.done = False
        self.current_task = task
        self.next_target = next_target
        self.progress = 0
        self.start_time = time.time()
        self.progress_type = progress_type
        self.directory = directory
        if progress_type == "detailed":
            t = threading.Thread(target=self.animate_detailed)
        else:
            t = threading.Thread(target=self.animate_general)
        t.start()
        return t

    def update(self, task, progress, next_target, total, progress_details=None):
        self.current_task = task
        self.progress = progress
        self.next_target = next_target
        self.progress_details = progress_details

    def stop(self, t):
        self.done = True
        t.join()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        task = sys.argv[1]
        next_target = sys.argv[2] if len(sys.argv) > 2 else "Unknown"
        progress_type = sys.argv[3] if len(sys.argv) > 3 else "general"
        directory = sys.argv[4] if len(sys.argv) > 4 else "."
    else:
        task = "Processing"
        next_target = "Unknown"
        progress_type = "general"
        directory = "."

    total = 100
    pb = ProgressBar()
    t = pb.start(task, next_target, progress_type, directory)
    for i in range(total):
        time.sleep(0.1)
        pb.update(task, i + 1, next_target, total)
    pb.stop(t)
