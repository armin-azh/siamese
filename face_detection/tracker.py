import time
from .filter import Kalman

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

    def add_new_tracker(self, id_name) -> KalmanFaceTracker:
        """
        add new id name to the list
        :param id_name:
        :return:
        """
        result = self.search(id_name)
        if result is not None:
            return result
        else:
            result = KalmanFaceTracker(initial_name=id_name)
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

    def update(self):
        """
        update tracker state
        :return:
        """
        self._modifier()

    # def modify_trackers
