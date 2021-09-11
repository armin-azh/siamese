from pathlib import Path

from v2.core.source import SourceProvider, FileSourceModel
from v2.core.distance import CosineDistanceV2, CosineDistanceV1
from v2.core.db import SimpleDatabase
from v2.core.network import (MultiCascadeFaceDetector,
                             FaceNetModel,
                             MaskModel,
                             HPEModel)
from v2.core.engine import RawVisualService, ClusteringService
from v2.tools.logger import LOG_Path
from settings import (CAMERA_MODEL_CONF,
                      BASE_DIR,
                      DETECTOR_CONF,
                      MODEL_CONF,
                      GALLERY_CONF,
                      DEFAULT_CONF,
                      MASK_CONF,
                      HPE_CONF)
from settings import (PATH_NORMAL,
                      PATH_MASK)


def clustering_v1_service(phase="normal") -> ClusteringService:
    base = Path(BASE_DIR)
    files = None
    if phase == "normal":
        files = list(PATH_NORMAL.glob("*"))
    elif phase == "mask":
        files = list(PATH_MASK.glob("*"))
    else:
        raise ValueError("wrong phase")

    n_cluster = int(GALLERY_CONF.get("n_clusters"))

    files_sources = [_file for _file in files if _file.is_file()]

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

    # db = SimpleDatabase(db_path=Path("G:\\Documents\\Project\\facerecognition\\data\\temp\\db"))
    hpe_conf = (
        float(HPE_CONF.get("pan_left")),
        float(HPE_CONF.get("pan_right")),
        float(HPE_CONF.get("tilt_up")),
        float(HPE_CONF.get("tilt_down"))
    )

    hpe = HPEModel(model_path=base.joinpath(HPE_CONF.get("model")),
                   img_norm=(float(HPE_CONF.get("im_norm_mean")), float(HPE_CONF.get("im_norm_var"))),
                   tilt_norm=(float(HPE_CONF.get("tilt_norm_mean")), float(HPE_CONF.get("tilt_norm_var"))),
                   pan_norm=(float(HPE_CONF.get("pan_norm_mean")), float(HPE_CONF.get("pan_norm_var"))),
                   rescale=float(HPE_CONF.get("rescale")),
                   conf=hpe_conf
                   )

    return ClusteringService(source_pool=files_sources,
                             face_detector=face_dm,
                             embedded=embedded_model,
                             database=db,
                             hpe=hpe,
                             log_path=LOG_Path,
                             n_cluster=n_cluster,
                             name="basic_raw_visualization_service")


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

    distance = CosineDistanceV1(similarity_threshold=float(DEFAULT_CONF.get("similarity_threshold")))

    mask_detector = MaskModel(model_path=base.joinpath(MASK_CONF.get("model")),
                              score_threshold=float(MASK_CONF.get("score_threshold")))

    hpe_conf = (
        float(HPE_CONF.get("pan_left")),
        float(HPE_CONF.get("pan_right")),
        float(HPE_CONF.get("tilt_up")),
        float(HPE_CONF.get("tilt_down"))
    )

    hpe = HPEModel(model_path=base.joinpath(HPE_CONF.get("model")),
                   img_norm=(float(HPE_CONF.get("im_norm_mean")), float(HPE_CONF.get("im_norm_var"))),
                   tilt_norm=(float(HPE_CONF.get("tilt_norm_mean")), float(HPE_CONF.get("tilt_norm_var"))),
                   pan_norm=(float(HPE_CONF.get("pan_norm_mean")), float(HPE_CONF.get("pan_norm_var"))),
                   rescale=float(HPE_CONF.get("rescale")),
                   conf=hpe_conf
                   )

    return RawVisualService(source_pool=vision_source(),
                            face_detector=face_dm,
                            embedded=embedded_model,
                            database=db,
                            distance=distance,
                            mask_detector=mask_detector,
                            hpe=hpe,
                            log_path=LOG_Path,
                            name="basic_raw_visualization_service")
