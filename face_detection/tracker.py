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
