import numpy as np
from filterpy.kalman import KalmanFilter


class Kalman:
    """
    initiate Kalman filter
    """

    def __init__(self, det):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = np.array([det[0], det[1], det[2], det[3]]).reshape((4, 1))
        super().__init__()

    def predict(self):
        """
        predict new coordinate with kalman filter
        :return:
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        return self.kf.x

    def correction(self, measurement):
        """
        do correction step on kalman filter
        :param measurement:
        :return:
        """
        self.kf.update(measurement)

    def get_current_state(self):
        """
        get current state of the kalman filter
        :return:
        """
        coordinate = (np.array([self.kf.x[0], self.kf.x[1], self.kf.x[2], self.kf.x[3]]).reshape((1, 4)))
        return coordinate
