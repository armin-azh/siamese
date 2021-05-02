import platform

import tensorflow as tf


def system_status(args):
    re = dict()
    re["Platform"] = platform.platform()
    re["Processor"] = platform.processor()

    devices = tf.config.list_physical_devices()
    for idx, dev in enumerate(devices):
        re[f"Device {idx + 1}"] = dev.name

    re["CUDA Enabled"] = tf.test.is_built_with_cuda()
    for key, value in re.items():
        print(f"$ {key}: {value}")
