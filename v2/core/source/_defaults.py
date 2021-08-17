from typing import Tuple
from pathlib import Path

from ._base import BaseSource
from v2.core.source._image import SourceImage

from .exceptions import *


class FileSource(BaseSource):
    def __init__(self, uuid: str, src: Path, output_size: Tuple[int, int], queue_size: int, logg_path: Path,
                 display: bool = False):
        if not src.exists():
            raise SourceIsNotExist("this source is not exists")
        if not src.is_file():
            raise SourceIsNotExist("this source is not a file")
        super(FileSource, self).__init__(uuid=uuid,
                                         src=str(src),
                                         src_type="file",
                                         output_size=output_size,
                                         queue_size=queue_size,
                                         logg_path=logg_path,
                                         display=display)

        self._finished = False

    @property
    def is_finished(self) -> bool:
        return self._finished

    def _get_frame(self):

        while True:
            try:
                if self._cap.isOpened() and self._online:
                    status, frame = self._cap.read()
                    if status:
                        self._frame_dequeue.append(SourceImage(im=frame))
                    else:
                        self._cap.release()
                        self._online = False
                        self._finished = True
                else:
                    self._finished = True
                self._spin(0.001)
            except AttributeError:
                pass

    def stream(self):
        if not self._online:
            self._spin(1)
            return None, None, self._finished

        if self._frame_dequeue and self._online:
            frame = self._frame_dequeue[-1].get_pixel
            return frame, self._convertor.normalize(mat=frame), self._finished, self._frame_dequeue[-1].timestamp
        else:
            return None, None, self._finished, self._frame_dequeue[-1].timestamp


class ProtocolSource(BaseSource):
    def __init__(self, uuid: str, src: str, output_size: Tuple[int, int], queue_size: int, logg_path: Path,
                 display: bool = False):
        super(ProtocolSource, self).__init__(uuid=uuid,
                                             src=src,
                                             src_type="protocol",
                                             output_size=output_size,
                                             queue_size=queue_size,
                                             logg_path=logg_path,
                                             display=display)


class WebCamSource(BaseSource):
    def __init__(self, uuid: str, src: int, output_size: Tuple[int, int], queue_size: int, logg_path: Path,
                 display: bool = False):
        super(WebCamSource, self).__init__(uuid=uuid,
                                           src=src,
                                           src_type="webCam",
                                           output_size=output_size,
                                           queue_size=queue_size,
                                           logg_path=logg_path,
                                           display=display)
