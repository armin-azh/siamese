from pathlib import Path
import yaml

from ._base import SourcePool
from ._defaults import *


class BaseProvider:
    def __init__(self, yam_path: Path, logg_path: Path):
        if not yam_path.exists() or not yam_path.is_file():
            raise FileNotFoundError("file not exists or the path is not a file")
        self._yaml_path = yam_path
        self._log_path = logg_path

    def _write(self, data: dict) -> None:
        with open(str(self._yaml_path), "w") as f:
            yaml.dump(data, f)

    def _read(self) -> dict:
        with open(str(self._yaml_path), "r") as f:
            data = yaml.safe_load(f)
        return data

    def __call__(self, *args, **kwargs) -> SourcePool:
        data = self._read()
        next_data = dict()
        cam_list = []

        cam_keys = data.keys()

        for cam in cam_keys:
            meta = data[cam]
            cam_type = meta.get("type")
            source = meta["source"]
            display = meta.get("display_log")
            if display is None:
                display = False
            q_size = meta.get("queue_size")
            if q_size is None:
                q_size = 10
            o_shape = (meta.get("output_size")[0], meta.get("output_size")[1]) if meta.get(
                "output_size") is not None else (640, 480)
            uu_id = meta.get("uuid")
            n = None

            if cam_type == "file":
                n = FileSource(uuid=uu_id, src=source, output_size=o_shape, queue_size=q_size, logg_path=self._log_path,
                               display=display)
            elif cam_type == "webcam":
                n = ProtocolSource(uuid=uu_id, src=source, output_size=o_shape, queue_size=q_size,
                                   logg_path=self._log_path,
                                   display=display)
            elif cam_type == "protocol":
                n = WebCamSource(uuid=uu_id, src=source, output_size=o_shape, queue_size=q_size,
                                 logg_path=self._log_path,
                                 display=display)

            data[cam] = n.get_id
            next_data[cam] = {"type": cam_type, "source": source, "display_log": display, "queue_size": q_size,
                              "output_size": [o_shape[0], o_shape[1]],"uuid":n.get_id}
            cam_list.append(n)

        self._write(next_data)

        return SourcePool(src_list=cam_list)
