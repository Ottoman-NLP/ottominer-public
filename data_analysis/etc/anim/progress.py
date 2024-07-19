
import threading
import time
import sys
import tkinter as tk
from itercycle import AnimatedProgress
class ProgressBar(AnimatedProgress):
    def __init__(self):
        super().__init__()
        self.progress_type = "general"
        self.window = None
        self.label = None
        self.progress_var = None

    def create_window(self):
        self.window = tk.Tk()
        self.window.title("Progress")
        self.label = tk.Label(self.window, text=f"{self.current_task} - {self.next_target}")
        self.label.pack()
        self.progress_var = tk.DoubleVar()
        progress_bar = tk.Progressbar(self.window, variable=self.progress_var, maximum=100)
        progress_bar.pack(fill=tk.X, expand=True)
        self.window.after(100, self.update_window)
        threading.Thread(target=self.window.mainloop).start()

    def update_window(self):
        if self.done:
            self.window.quit()
            return
        self.label.config(text=f"{self.current_task} - {self.next_target}")
        self.progress_var.set(self.progress)
        self.window.after(100, self.update_window)

    def start(self, task, next_target, progress_type="general", directory=""):
        self.done = False
        self.current_task = task
        self.next_target = next_target
        self.progress = 0
        self.start_time = time.time()
        self.progress_type = progress_type
        self.directory = directory
        self.create_window()
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
        if progress_details is None:
            try:
                from itercycle import get_system_metrics
            except ImportError:
                pass
            else:
                progress_details = get_system_metrics(progress, total, self.start_time, self.directory)
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
