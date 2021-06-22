import imagezmq
import socket
from stream.source import OpencvSource


class CameraNode:
    def __init__(self, src, name, width, height):
        self._vid_source = OpencvSource(src, name, width, height)
        self._host_name = socket.gethostname()
        self._msg = self._host_name + "_" + self._vid_source.name

    def run(self):
        sender = imagezmq.ImageSender(connect_to="tcp://127.0.0.1:5555")

        while self._vid_source.isOpened():
            _, frame = self._vid_source.read()
            sender.send_image(self._msg, frame)
