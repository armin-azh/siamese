import time
from filter import Kalman


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

    def __init__(self, initial_name):
        self._tk_cnt = TrackerCounter()
        self._id_name = initial_name
        self._modified = time.time()

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

    def modify(self) -> None:
        """
        modify the face tracker instance
        :return:
        """
        self._modified = time.time()


class KalmanFaceTracker(FaceTracker, Kalman):

    def __init__(self, initial_name):
        super().__init__(initial_name)
