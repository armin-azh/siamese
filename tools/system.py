from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import platform


import tensorflow as tf
from tensorflow.python.client import device_lib


def system_status(args):
    re = dict()
    devices_ = device_lib.list_local_devices()
    re["Platform"] = platform.platform()
    re["Processor"] = platform.processor()
    for i in devices_:
        if i.device_type == "CPU":
<<<<<<< HEAD
            re["Processor_MEMORY_LIMIT"] = i.memory_limit
=======
            re["Processor_MEMORY_LIMIT"]=i.memory_limit
>>>>>>> e84ed99257ff9b213c0eb9b7b90df6ce0f3f719c

    idx = 0
    for dev in devices_:
        if dev.device_type == "GPU":
<<<<<<< HEAD
            prefix = "GPU_" + str(idx)
            re[prefix + "_NAME"] = dev.physical_device_desc
            re[prefix + "_MEMORY_LIMIT"] = dev.memory_limit
            idx += 1
=======
            prefix = "GPU_"+str(idx)
            re[prefix+"_NAME"] = dev.physical_device_desc
            re[prefix + "_MEMORY_LIMIT"]=dev.memory_limit
            idx+=1
>>>>>>> e84ed99257ff9b213c0eb9b7b90df6ce0f3f719c

    re["CUDA Enabled"] = tf.test.is_built_with_cuda()
    for key, value in re.items():
        print(f"$ {key}: {value}")

<<<<<<< HEAD
    return re
=======


system_status('ar')
>>>>>>> e84ed99257ff9b213c0eb9b7b90df6ce0f3f719c
