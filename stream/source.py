import subprocess
from threading import Thread
import re
import numpy as np
import cv2
from v2.core.source import SourceProvider
from v2.tools.logger import LOG_Path
from settings import CAMERA_MODEL_CONF, BASE_DIR
from pathlib import Path


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
    def __init__(self, src, name, width, height, *args, **kwargs):
        self._name = name
        self._src = src
        self._size = (width, height)
        self._origin_frame = None
        self._origin_size = None
        self._source = cv2.VideoCapture(self._src)
        super(OpencvSource, self).__init__(*args, **kwargs)

    def read(self):
        ret, frame = self._source.read()
        # print(frame.dtype)
        self._origin_frame = frame.copy() if frame is not None else None
        self._origin_size = frame.shape[:2] if frame is not None else None
        frame = cv2.resize(frame, self._size).astype(np.uint8) if frame is not None else frame

        return ret, frame

    def isOpened(self) -> bool:
        return self._source.isOpened()

    def release(self) -> None:
        self._source.release()

    def get(self, att):
        return self._source.get(att)

    @property
    def scale_factor(self):
        if self._origin_size is None:
            return None, None

        else:
            x_s = self._origin_size[1] / self._size[0]
            y_s = self._origin_size[0] / self._size[1]
            return x_s, y_s

    def convert_coordinate(self, b_box, margin=(0, 0)):
        x_s, y_s = self.scale_factor
        x1, y1, x2, y2 = b_box[0], b_box[1], b_box[2], b_box[3]
        return self.margin([int(x1 * x_s), int(y1 * y_s), int(x2 * x_s), int(y2 * y_s)], margin[0], margin[0])

    @property
    def original_frame(self) -> np.ndarray:
        return self._origin_frame

    @property
    def name(self) -> str:
        return self._name

    def margin(self, b_box, x_margin, y_margin):

        x1, y1, x2, y2 = b_box[0], b_box[1], b_box[2], b_box[3]

        x1 = x1 - x_margin
        y1 = y1 - y_margin
        x2 = x2 + x_margin
        y2 = y2 + y_margin

        x1_ = np.maximum(x1, 0)
        y1_ = np.maximum(y1, 0)
        x2_ = np.minimum(x2, self._origin_size[1])
        y2_ = np.minimum(y2, self._origin_size[0])

        return x1_, y1_, x2_, y2_


class MultiSource(Source):
    def __init__(self, src, name, width, height, *args, **kwargs):
        self._name = name
        self._src = src
        self._size = (width, height)
        self._origin_frame = None
        self._origin_size = None
        base = Path(BASE_DIR)
        self._source = SourceProvider(logg_path=LOG_Path, yam_path=base.joinpath(CAMERA_MODEL_CONF.get("conf")))()
        super(MultiSource, self).__init__(*args, **kwargs)

    def read(self):
        origin_frame, frame,cap_id,time_stamp = self._source.next_stream()
        # print(frame.dtype)
        self._origin_frame = origin_frame.copy() if origin_frame is not None else None
        self._origin_size = origin_frame.shape[:2] if origin_frame is not None else None

        ret = None if frame is None else True
        return ret, frame

    # def isOpened(self) -> bool:
    #     return self._source.isOpened()
    #
    # def release(self) -> None:
    #     self._source.release()
    #
    # def get(self, att):
    #     return self._source.get(att)

    @property
    def scale_factor(self):
        if self._origin_size is None:
            return None, None

        else:
            x_s = self._origin_size[1] / self._size[0]
            y_s = self._origin_size[0] / self._size[1]
            return x_s, y_s

    def convert_coordinate(self, b_box, margin=(0, 0)):
        x_s, y_s = self.scale_factor
        x1, y1, x2, y2 = b_box[0], b_box[1], b_box[2], b_box[3]
        return self.margin([int(x1 * x_s), int(y1 * y_s), int(x2 * x_s), int(y2 * y_s)], margin[0], margin[0])

    @property
    def original_frame(self) -> np.ndarray:
        return self._origin_frame

    @property
    def name(self) -> str:
        return self._name

    def margin(self, b_box, x_margin, y_margin):

        x1, y1, x2, y2 = b_box[0], b_box[1], b_box[2], b_box[3]

        x1 = x1 - x_margin
        y1 = y1 - y_margin
        x2 = x2 + x_margin
        y2 = y2 + y_margin

        x1_ = np.maximum(x1, 0)
        y1_ = np.maximum(y1, 0)
        x2_ = np.minimum(x2, self._origin_size[1])
        y2_ = np.minimum(y2, self._origin_size[0])

        return x1_, y1_, x2_, y2_


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
