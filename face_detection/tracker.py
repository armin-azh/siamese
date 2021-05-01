import time
import numpy as np
from .filter import Kalman
from .utils import bulk_calculate_iou

from settings import TRACKER_CONF


class TrackerCounter:
    """
    a class for consider tracker counter
    """
    init_track_id = 1

    def __init__(self):
        self.track_counter = 1
        self.track_id = self.init_track_id
        TrackerCounter.next_track_id()

    @classmethod
    def next_track_id(cls):
        cls.init_track_id += 1

    def __call__(self):
        self.track_counter += 1

    @property
    def counter(self):
        return self.track_counter


class FaceTracker:
    STATUS_MATCHED = 'matched'
    STATUS_UNMATCHED = 'unmatched'

    def __init__(self, initial_name, **kwargs):
        self._tk_cnt = TrackerCounter()
        self._id_name = initial_name
        self._modified = time.time()
        self._status = self.STATUS_UNMATCHED
        super().__init__(**kwargs)

    @property
    def name(self):
        return self._id_name

    @name.setter
    def name(self, n_name):
        self._id_name = n_name

    @property
    def face_id(self) -> int:
        return self._tk_cnt.track_id

    def __call__(self, n_name=None) -> None:
        if n_name is not None:
            self._id_name = n_name
        self._tk_cnt()  # increase detected counter
        self.modify()

    def modify(self) -> None:
        """
        modify the face tracker instance
        :return:
        """
        self._modified = time.time()
        if self._tk_cnt.counter == int(TRACKER_CONF.get("max_frame_conf")):
            self._status = self.STATUS_MATCHED

    @property
    def last_modified(self):
        return self._modified

    @property
    def counter(self):
        return self._tk_cnt.counter

    @property
    def status(self):
        return self._status


class KalmanFaceTracker(FaceTracker, Kalman):

    def __init__(self, initial_name, det):
        super().__init__(initial_name=initial_name, det=det)


class Tracker:
    """
    management class for trackers
    """
    _global_time = time.time()

    def __init__(self):
        self._in_memory_tk_faces = []
        self._max_keep_tk_sec = int(TRACKER_CONF.get("kalman_max_save_tk_sec"))

    @property
    def global_time(self):
        return self._global_time

    def _update_global_time(self) -> None:
        """
        this method should be call on every main operation that the class do
        :return:
        """
        self._global_time = time.time()

    def _refactor_in_memory_tk(self):
        """
        delete instances that force _max_keep_tk_sec
        :return:
        """
        if self._in_memory_tk_faces:
            for idx, tk_face in enumerate(self._in_memory_tk_faces):
                if abs(self._global_time - tk_face.last_modified) > self._max_keep_tk_sec:
                    self._in_memory_tk_faces.remove(self._in_memory_tk_faces[idx])

    def _modifier(self) -> None:
        """
        modify global time and refactor tracker list
        :return:
        """
        self._update_global_time()
        self._refactor_in_memory_tk()

    def add_new_tracker(self, id_name, coordinate) -> KalmanFaceTracker:
        """
        add new id name to the list
        :param coordinate:
        :param id_name:
        :return:
        """
        result = self.search(id_name)
        if result is not None:
            if result.status == KalmanFaceTracker.STATUS_UNMATCHED:
                result.correction(coordinate)
            self.modify_tracker(id_name)
            return result
        else:
            result = KalmanFaceTracker(initial_name=id_name, det=coordinate)
            self._in_memory_tk_faces.append(result)
            return result

    def search(self, id_name):
        """
        search a tracker with identity name
        :param id_name:
        :return:
        """

        for tk in self._in_memory_tk_faces:
            if tk.name == id_name:
                return tk
        return None

    def modify_tracker(self, id_name):
        result = self.search(id_name)
        if result is not None:
            result()

    def _split_trackers(self):
        """
        split the tracker in to which satisfies the condition
        :return:
        """
        satisfied = list()
        n_satisfied = list()

        for tk in self._in_memory_tk_faces:
            if tk.status == FaceTracker.STATUS_MATCHED:
                satisfied.append(tk)
            else:
                n_satisfied.append(tk)

        return satisfied, n_satisfied

    def update(self):
        """
        update tracker state
        :return:
        """
        self._modifier()

    def grab_satisfied_trackers(self):
        """
        find trackers that are satisfied more than the threshold
        :return: generator
        """
        for tk in self._in_memory_tk_faces:
            if tk.status == KalmanFaceTracker.STATUS_MATCHED:
                yield tk

    def find_relative_boxes(self, detections):
        """
        find match and un-matched boxes
        :param detections: matrix in shape (m,4)
        :return: matches and un-matches list
        """

        boxes = []
        for tk in self.grab_satisfied_trackers():
            tk.predict()
            boxes.append(tk.get_current_state()[0])

        boxes = np.array(boxes)
        iou_matrix = bulk_calculate_iou(detections, boxes)
        if iou_matrix.shape[0] > 0 and iou_matrix.shape[1] > 0:
            max_iou = np.max(iou_matrix, axis=1)
            max_args_iou = np.argmax(iou_matrix, axis=1)
            un_matches = np.where(max_iou < float(TRACKER_CONF.get("iou_threshold")))
            matches = np.where(max_iou >= float(TRACKER_CONF.get("iou_threshold")))
            # print(matches)
            # print(len(matches))
            if len(matches) > 0:
                matches = matches[0][max_args_iou]
            else:
                matches = None
            return un_matches, matches
        else:
            return None, None

    @property
    def number_of_trackers(self):
        return len(self._in_memory_tk_faces)

    def retrieve_trackers_by_index(self, indexes):
        for idx, tk in enumerate(self._in_memory_tk_faces):
            if idx in indexes:
                yield tk

    def get_tracker_current_state(self,key):
        return self._in_memory_tk_faces[key].get_current_state()