import psutil
import os
import hashlib
import time

def get_folder_size(directory):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

def simple_hash(data):
    return int(hashlib.md5(data.encode()).hexdigest(), 16)

def get_system_metrics(progress, total, start_time, directory):
    try:
        import cupy as cp
        gpu_usage = cp.cuda.Device(0).use().memoryInfo().free / cp.cuda.Device(0).use().memoryInfo().total * 100
    except Exception:
        gpu_usage = 0

    elapsed_time = time.time() - start_time
    eta = (total - progress) * (elapsed_time / progress) if progress > 0 else 0
    folder_size = get_folder_size(directory) / (1024 * 1024)  # Convert to MB

    system_metrics = {
        'gpu': gpu_usage,
        'cpu': psutil.cpu_percent(),
        'switches': psutil.cpu_stats().ctx_switches,
        'fs_in': psutil.disk_io_counters().read_count,
        'fs_out': psutil.disk_io_counters().write_count,
        'size': folder_size,
        'hash': simple_hash(str(progress) + str(total)),
        'eta': eta
    }

    return system_metrics

