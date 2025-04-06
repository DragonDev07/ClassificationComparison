import time

import numpy as np
import psutil


class ResourceMonitor:
    def __init__(self):
        self.memory_usage = []
        self.cpu_usage = []
        self.start_time = None

    # `start()` -->
    #   - Start monitoring system resources
    def start(self):
        self.start_time = time.time()
        self.memory_usage = []
        self.cpu_usage = []

    # `_collect_metrics()` -->
    #   - Collect current system metrics
    def _collect_metrics(self):
        process = psutil.Process()
        self.memory_usage.append(process.memory_info().rss / 1024 / 1024)  # MB
        self.cpu_usage.append(psutil.cpu_percent())

    # `stop()` -->
    #   - Stop monitoring and return average metrics
    def stop(self):
        return {
            "memory_usage": np.mean(self.memory_usage),
            "cpu_usage": np.mean(self.cpu_usage),
        }
