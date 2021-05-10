from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import platform
import psutil
import tensorflow as tf
from tensorflow.python.client import device_lib


def system_status(args, printed=True):
    re = dict()
    devices_ = device_lib.list_local_devices()
    re["Platform"] = platform.platform()
    re["Processor"] = platform.processor()
    re["Processor_CORES"] = psutil.cpu_count()
    re["Processor_TOTAL_MEMORY"] = str(psutil.virtual_memory().total / 1024 ** 2) + " MB"
    re["Processor_USED_MEMORY"] = str(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2) + " MB"
    for i in devices_:
        if i.device_type == "CPU":
            re["Processor_MEMORY_LIMIT"] = str(i.memory_limit / 1024 ** 2) + " MB"

    idx = 0
    for dev in devices_:
        if dev.device_type == "GPU":
            prefix = "GPU_" + str(idx)
            re[prefix + "_NAME"] = dev.physical_device_desc
            re[prefix + "_MEMORY_LIMIT"] = str(dev.memory_limit / 1024 ** 2) + " MB"
            idx += 1

    re["CUDA Enabled"] = tf.test.is_built_with_cuda()
    if printed:
        for key, value in re.items():
            print(f"$ {key}: {value}")

    return re
