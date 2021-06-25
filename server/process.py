import subprocess
import imagezmq
import zmq
from stream.source import OpencvSource
import socket
import cv2
from datetime import datetime
import sys

from tools.logger import Logger
from settings import ZERO_MQ_CONF


class BasicProcess:
    def __init__(self, name, *args, **kwargs):
        self._p_name = name

        super().__init__(*args, **kwargs)

    def run(self):
        raise NotImplementedError("This method should implemented with child class")


class MainHubNodeProcess(BasicProcess):
    def __init__(self, *args, **kwargs):
        self._im_hub = imagezmq.ImageHub()

        super(MainHubNodeProcess, self).__init__("main_hub_process", *args, **kwargs)

    def run(self):
        logger = Logger()
        print("$ Main Hub is now starting")

        while True:
            _d_format = f"[{datetime.strftime(datetime.now(), '%Y-%m-%d : %H-%M-%S')}]"
            try:
                msg, frame = self._im_hub.recv_image()
                cv2.imshow('Main', frame)
                print(_d_format+" Received from " + msg)
                cv2.waitKey(1)
                self._im_hub.send_reply(b'Ok')

            except zmq.error.ZMQError:
                print(_d_format+" can`t connect")

            except KeyboardInterrupt:
                print(_d_format+" terminated")

            # image_hub.send_reply(b'Ok')


class CameraNodeProcess(BasicProcess):
    def __init__(self, src, name, width, height, *args, **kwargs):
        self._ip = ZERO_MQ_CONF.get("camera1_node")
        self._sender = imagezmq.ImageSender(connect_to="tcp://127.0.0.1:5555")
        self._vid_source = OpencvSource(src, name, width, height)
        super(CameraNodeProcess, self).__init__(socket.gethostname() + "_Cam1", *args, **kwargs)

    def run(self):
        logger = Logger()
        logger.warn("Camera 1 is now starting")
        while self._vid_source.isOpened():
            _, frame = self._vid_source.read()
            logger.info("Sent", timestamp=True)
            self._sender.send_image(self._p_name, frame)

        logger.warn("Camera Process Closed.")


class TensorflowProcess(BasicProcess):
    def __init__(self, *args, **kwargs):
        super(TensorflowProcess, self).__init__(*args, **kwargs)

    def run(self):
        pass


if __name__ == "__main__":
    cam = CameraNodeProcess(0, "rtsp_camera", 640, 480)
    hub = MainHubNodeProcess()

    cam.run()
    hub.run()
