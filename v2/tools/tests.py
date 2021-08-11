import time
from unittest import TestCase
from ._measurement import Counter, FPS


class CounterTestCase(TestCase):
    def test_counter_cnt(self):
        counter = Counter()

        for _ in range(60):
            counter.next()

        self.assertEqual(counter(), 60)

    def test_counter_with_limit(self):
        counter = Counter(limit=60)
        for _ in range(60):
            counter.next()
        self.assertEqual(counter(), 0)
        counter.next()
        self.assertEqual(counter(), 1)

    def test_reset_counter(self):
        counter = Counter()
        for _ in range(60):
            counter.next()
        counter.reset()
        self.assertEqual(counter(), 0)


class FpsTestCase(TestCase):
    def test_create_fps(self):
        fps = FPS()
        fps.start()
        fps.update()
        time.sleep(3)
        fps.stop()
        self.assertTrue(fps.fps() > 0)

    def test_create_fps_realtime(self):
        fps = FPS()
        fps.start()
        fps.update()
        fps.stop()
        self.assertEqual(fps.fps(), 0.)
