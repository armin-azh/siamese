import subprocess
from threading import Thread
import re
import numpy as np
import cv2


class Source:
    def __init__(self, *args, **kwargs):
        super(Source, self).__init__(*args, **kwargs)

    def isOpened(self):
        raise NotImplementedError

    def release(self):
        raise NotImplementedError

    def read(self):
        raise NotImplementedError


class OpencvSource(Source):
    def __init__(self, src, name, *args, **kwargs):
        self._name = name
        self._src = src
        self._source = cv2.VideoCapture(self._src)
        super(OpencvSource, self).__init__(*args, **kwargs)

    def read(self):
        return self._source.read()

    def isOpened(self) -> bool:
        return self._source.isOpened()

    def release(self) -> None:
        self._source.release()

    @property
    def name(self) -> str:
        return self._name


class RtspSource(Source):
    RGB = 'rgb24'
    GRAY = 'gray'
    BGR = '"bgr24"'

    def __init__(self, url: str, width: int, height: int, color: str = "bgr24", *args, **kwargs):
        self._url = url
        self._dim = (height, width, 3) if color == self.BGR or color == self.RGB else (height, width)
        self._fs = self._dim[0] * self._dim[1] * self._dim[2] if len(self._dim) == 3 else self._dim[0] * self._dim[1]
        self._color = color
        instruction = [
            'ffmpeg',
            '-i', self._url,
            '-f', 'rawvideo',  # Use image2pipe demuxer
            '-pix_fmt', self._color,  # Set BGR pixel format
            '-video_size', "640x480",
            '-vcodec', 'libx264',
            '-crf', '0',

            '-'
        ]
        # instruction = ['ffmpeg',"-i", self._url, "-f", "rawvideo", "-pix_fmt", self._color, "-"]
        self._process = subprocess.Popen(instruction, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        super(RtspSource, self).__init__(*args, **kwargs)

    def isOpened(self):
        pass

    def read(self):

        if self._process.poll():
            return False, None

        o = self._process.stdout.read(self._fs)
        print(o)

        if o == "":
            return False, None

        else:
            return True, np.frombuffer(o, dtype=np.uint8).reshape(self._dim)

    def _validate_url(self):
        pass


class WebcamVideoStream:
    def __init__(self, src=0, name="WebcamVideoStream"):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the thread name
        self.name = name

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


class VideoStream:
    def __init__(self, src=0, resolution=(320, 240),
                 framerate=32, **kwargs):
        self.stream = WebcamVideoStream(src=src)

    def start(self):
        # start the threaded video stream
        return self.stream.start()

    def update(self):
        # grab the next frame from the stream
        self.stream.update()

    def read(self):
        # return the current frame
        return self.stream.read()

    def stop(self):
        # stop the thread and release any resources
        self.stream.stop()
