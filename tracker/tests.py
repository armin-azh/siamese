import unittest
import time
from .timer import Timer


class TrackerTestCase(unittest.TestCase):

    def test_timer(self):
        threshold = 3.0
        timer = Timer(threshold=threshold)
        time.sleep(threshold)
        self.assertFalse(timer.validate())


if __name__ == "__main__":
    unittest.main()
