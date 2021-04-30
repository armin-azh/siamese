import unittest
from .tracker import *
from settings import TRACKER_CONF


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
        tk = KalmanFaceTracker(initial_name=ts_name)
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


if __name__ == "__main__":
    unittest.main()
