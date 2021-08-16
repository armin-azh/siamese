from pathlib import Path

from v2.core.source import SourceProvider
from v2.core.distance import CosineDistanceV2
from v2.core.db import SimpleDatabase
from v2.core.network import (MultiCascadeFaceDetector,
                             FaceNetModel)
from v2.core.engine import RawVisualService
from v2.tools.logger import LOG_Path
from settings import (CAMERA_MODEL_CONF,
                      BASE_DIR,
                      DETECTOR_CONF,
                      MODEL_CONF,
                      GALLERY_CONF,
                      DEFAULT_CONF)


def raw_visual_v1_service() -> RawVisualService:
    base = Path(BASE_DIR)
    vision_source = SourceProvider(logg_path=LOG_Path, yam_path=base.joinpath(CAMERA_MODEL_CONF.get("conf")))

    minsize = int(DETECTOR_CONF.get("min_face_size"))
    threshold = [
        float(DETECTOR_CONF.get("step1_threshold")),
        float(DETECTOR_CONF.get("step2_threshold")),
        float(DETECTOR_CONF.get("step3_threshold"))]
    factor = float(DETECTOR_CONF.get("scale_factor"))
    face_dm = MultiCascadeFaceDetector(stages_threshold=threshold, scale_factor=factor, min_face=minsize,
                                       name="raw_face_detector")

    embedded_model = FaceNetModel(model_path=base.joinpath(MODEL_CONF.get('facenet')))

    db = SimpleDatabase(db_path=base.joinpath(GALLERY_CONF.get("database_path")))

    distance = CosineDistanceV2(similarity_threshold=float(DEFAULT_CONF.get("similarity_threshold")))

    return RawVisualService(source_pool=vision_source(),
                            face_detector=face_dm,
                            embedded=embedded_model,
                            database=db,
                            distance=distance,
                            log_path=LOG_Path,
                            name="basic_raw_visualization_service")
