import unittest
import time
from .timer import Timer
from .tk import IdentityTracker, TrackerList
from .policy import Policy


class TrackerTestCase(unittest.TestCase):

    def test_policy(self):
        name = "armin"
        max_life_time = 3.0
        max_conf = 5
        a = Policy(name, max_life_time, max_conf)

        for i in range(3):
            a()
            self.assertEqual(a.status, Policy.STATUS_NOT_CONF)

        time.sleep(2)

        a()
        self.assertEqual(a.status, Policy.STATUS_NOT_CONF)

        a()
        self.assertEqual(a.status, Policy.STATUS_CONF)

        time.sleep(4)

        a()
        self.assertEqual(a.status, Policy.STATUS_EXPIRED)

        a()

        self.assertEqual(a.status,Policy.STATUS_EXPIRED)

    def test_timer(self):
        threshold = 3.0
        timer = Timer(threshold=threshold)
        time.sleep(threshold)
        self.assertFalse(timer.validate())

    def test_identity_tracker(self):
        id_name = 'max'
        max_conf = 5
        timer_thresh = 3.4
        tk = IdentityTracker(name=id_name, max_time=timer_thresh, max_conf=max_conf)
        self.assertTrue(tk())
        self.assertTrue(tk.status == IdentityTracker.UN_MATCHED)

        for i in range(max_conf):
            tk()
        self.assertEqual(tk.status, IdentityTracker.MATCHED)
        time.sleep(timer_thresh)
        self.assertFalse(tk())

    def test_tracker_list(self):
        max_conf = 5
        timer_thresh = 3.4
        tk = TrackerList(timer_thresh, max_conf)
        res = tk("armin")

        self.assertEqual(res, IdentityTracker.UN_MATCHED)
        tk.modify()
        res = tk("armin")

        self.assertEqual(res, IdentityTracker.UN_MATCHED)

        for i in range(max_conf):
            tk("armin")

        res = tk("armin")
        self.assertEqual(res, IdentityTracker.MATCHED)
        time.sleep(timer_thresh)

        res = tk("armin")
        self.assertEqual(res, IdentityTracker.MATCHED)


if __name__ == "__main__":
    unittest.main()
