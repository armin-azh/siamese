from pathlib import Path

from settings import (CAMERA_MODEL_CONF,
                      BASE_DIR,
                      DETECTOR_CONF,
                      MODEL_CONF,
                      GALLERY_CONF,
                      DEFAULT_CONF,
                      MASK_CONF,
                      HPE_CONF,
                      TRACKER_CONF,
                      SERVER_CONF)
from settings import (PATH_NORMAL,
                      PATH_MASK)
from v2.core.db import SimpleDatabase
from v2.core.distance import CosineDistanceV2
from v2.core.engine import RawVisualService, ClusteringService, UDPService, RawVisualMahalanobisService
from v2.core.network import (MultiCascadeFaceDetector,
                             FaceNetModel,
                             MaskModel,
                             HPEModel)
from v2.core.source import SourceProvider
from v2.tools.logger import LOG_Path


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

    distance = CosineDistanceV2(similarity_threshold=float(DEFAULT_CONF.get("similarity_threshold")))

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

    trk_conf = {
        "face_threshold": float(DETECTOR_CONF.get("conf_thresh")),
        "detect_interval": int(DETECTOR_CONF.get("mtcnn_per_frame")),
        "iou_threshold": float(TRACKER_CONF.get("iou_threshold")),
        "max_age": int(TRACKER_CONF.get("max_age")),
        "min_hit": int(TRACKER_CONF.get("min_hits"))
    }

    return RawVisualService(source_pool=vision_source(),
                            face_detector=face_dm,
                            embedded=embedded_model,
                            database=db,
                            distance=distance,
                            mask_detector=mask_detector,
                            hpe=hpe,
                            tracker_conf=trk_conf,
                            log_path=LOG_Path,
                            name="basic_raw_visualization_service")


def udp_v1_service() -> UDPService:
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

    trk_conf = {
        "face_threshold": float(DETECTOR_CONF.get("conf_thresh")),
        "detect_interval": int(DETECTOR_CONF.get("mtcnn_per_frame")),
        "iou_threshold": float(TRACKER_CONF.get("iou_threshold")),
        "max_age": int(TRACKER_CONF.get("max_age")),
        "min_hit": int(TRACKER_CONF.get("min_hits"))
    }

    sock_conf = {
        "ip": SERVER_CONF.get("UDP_HOST"),
        "port": int(SERVER_CONF.get("UDP_PORT"))
    }

    pol_conf = {
        "max_life_time": float(TRACKER_CONF.get("recognized_max_modify_time")),
        "max_confidence_rec": int(TRACKER_CONF.get("recognized_max_frame_conf")),
        "max_confidence_un_rec": int(TRACKER_CONF.get("unrecognized_max_frame_conf"))
    }

    margin = (int(DETECTOR_CONF.get("x_margin")), int(DETECTOR_CONF.get("y_margin")))

    face_save_path = Path(SERVER_CONF.get("face_save_path")).joinpath(SERVER_CONF.get("face_folder"))
    if not face_save_path.exists():
        face_save_path.mkdir(parents=True)

    return UDPService(source_pool=vision_source(),
                      face_detector=face_dm,
                      embedded=embedded_model,
                      database=db,
                      distance=distance,
                      mask_detector=mask_detector,
                      hpe=hpe,
                      face_path=face_save_path,
                      tracker_conf=trk_conf,
                      policy_conf=pol_conf,
                      socket_conf=sock_conf,
                      margin=margin,
                      log_path=LOG_Path,
                      name="udp_service")


def raw_visual_v2_service() -> RawVisualMahalanobisService:
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

    trk_conf = {
        "face_threshold": float(DETECTOR_CONF.get("conf_thresh")),
        "detect_interval": int(DETECTOR_CONF.get("mtcnn_per_frame")),
        "iou_threshold": float(TRACKER_CONF.get("iou_threshold")),
        "max_age": int(TRACKER_CONF.get("max_age")),
        "min_hit": int(TRACKER_CONF.get("min_hits"))
    }

    return RawVisualMahalanobisService(source_pool=vision_source(),
                                       face_detector=face_dm,
                                       embedded=embedded_model,
                                       database=db,
                                       distance=distance,
                                       mask_detector=mask_detector,
                                       hpe=hpe,
                                       tracker_conf=trk_conf,
                                       log_path=LOG_Path,
                                       name="basic_raw_visualization_mahalanobis_service")
