import os
import hashlib
import logging
import psutil
from cupy.cuda import Device

class SysMetrics:
    def __init__(self):
        self.metrics = {
            'gpu_usage': 0,    # GPU usage as a percentage
            'cpu_usage': 0,    # CPU usage as a percentage
            'ctx_switches': 0, # Number of context switches
            'fs_read': 0,      # Filesystem reads in MB
            'fs_write': 0,     # Filesystem writes in MB
            'file_size': 0,    # File size in MB
            'hash_value': 0,   # Hash value of the directory
            'estimated_time': 0 # Estimated time to complete the task
        }
        self.directory = ""

    def get_system_metrics(self, directory):
        self.directory = directory
        self.update_cpu_gpu_usage()
        self.update_file_io()
        self.update_directory_metrics()

    def update_cpu_gpu_usage(self):
        try:
            self.metrics['gpu_usage'] = Device(0).mem_info[0] / Device(0).mem_info[1] * 100
            self.metrics['cpu_usage'] = psutil.cpu_percent(interval=1)
        except Exception as e:
            logging.error("Failed to get CPU/GPU usage: %s", e)

    def update_file_io(self):
        try:
            io_counters = psutil.disk_io_counters()
            self.metrics['fs_read'] = io_counters.read_bytes / (1024 ** 2)  # Convert bytes to MB
            self.metrics['fs_write'] = io_counters.write_bytes / (1024 ** 2)  # Convert bytes to MB
        except Exception as e:
            logging.error("Failed to get disk IO: %s", e)

    def update_directory_metrics(self):
        self.metrics['file_size'] = self.get_directory_size(self.directory) / (1024 ** 2)  # Convert bytes to MB
        self.metrics['hash_value'] = self.get_directory_hash(self.directory)

    def get_directory_size(self, path):
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if os.path.exists(fp):
                        total_size += os.path.getsize(fp)
        except Exception as e:
            logging.error("Failed to calculate directory size: %s", e)
        return total_size

    def get_directory_hash(self, path):
        hash_md5 = hashlib.md5()
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.isfile(filepath):
                        with open(filepath, "rb") as f:
                            for chunk in iter(lambda: f.read(4096), b""):
                                hash_md5.update(chunk)
            return int(hash_md5.hexdigest(), 16)
        except Exception as e:
            logging.error("Failed to hash directory: %s", e)
            return 0

if __name__ == "__main__":
    sm = SysMetrics()
    sm.get_system_metrics("/path/to/directory")  # Example path
    print(sm.metrics)
