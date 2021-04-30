import unittest
from .tracker import *
from settings import TRACKER_CONF
import time
import numpy as np


class FilterTest(unittest.TestCase):

    def test_tracker_counter(self):
        tk = TrackerCounter()
        self.assertEqual(tk.track_id, 1)
        self.assertEqual(tk.counter, 1)
        self.assertEqual(TrackerCounter.init_track_id, 2)

        for i in range(2, 10):
            tk()
            self.assertEqual(tk.counter, i)

    def test_face_tracker(self):
        ts_name = "Armin Azhdehnia"
        tk = FaceTracker(initial_name=ts_name)
        self.assertEqual(tk.name, ts_name)

        self.assertEqual(tk.face_id, 1)
        self.assertEqual(tk.counter, 1)

        for i in range(2, 8):
            tk()
            self.assertEqual(tk.counter, i)
            if tk.counter < int(TRACKER_CONF.get("max_frame_conf")):
                self.assertEqual(tk.status, FaceTracker.STATUS_UNMATCHED)
            else:
                self.assertEqual(tk.status, FaceTracker.STATUS_MATCHED)

    def test_kalman_face_tracker(self):
        ts_name = "Armin Azhdehnia"
        ts_coordinate = np.array([337, 12, 22, 44])
        tk = KalmanFaceTracker(initial_name=ts_name,det=ts_coordinate)
        self.assertEqual(tk.name, ts_name)

        self.assertEqual(tk.face_id, 1)
        self.assertEqual(tk.counter, 1)

        for i in range(2, 8):
            tk()
            self.assertEqual(tk.counter, i)
            if tk.counter < int(TRACKER_CONF.get("max_frame_conf")):
                self.assertEqual(tk.status, FaceTracker.STATUS_UNMATCHED)
            else:
                self.assertEqual(tk.status, FaceTracker.STATUS_MATCHED)

        ts_coordinate = np.array([114, 122, 32, 48])

        tk.correction(ts_coordinate)



    def test_tracker(self):
        tk = Tracker()
        self.assertGreaterEqual(time.time(), tk.global_time)

        ts_name = "David Cage"
        tk.add_new_tracker(id_name=ts_name)

        self.assertEqual(tk.search(id_name=ts_name).name, ts_name)
        time.sleep(int(TRACKER_CONF.get("kalman_max_save_tk_sec")))
        tk.update()
        self.assertEqual(tk.search(id_name=ts_name), None)

        ts_name = "Shultz"
        tk.add_new_tracker(ts_name)
        time.sleep(int(TRACKER_CONF.get("kalman_max_save_tk_sec")))
        tk.modify_tracker(ts_name)
        tk.update()
        self.assertEqual(tk.search(id_name=ts_name).name, ts_name)

    def test_tracker_main_functionality(self):
        pass


if __name__ == "__main__":
    unittest.main()
