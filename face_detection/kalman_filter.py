import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
import tensorflow as tf
from detector import FaceDetector
from recognition.utils import FPS
from sklearn.utils.linear_assignment_ import linear_assignment


def iou(bb_test, bb_gt):
    """
    Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return o


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """(numpy.array, numpy.array, int) -> numpy.array, numpy.array, numpy.array
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """

    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 4), dtype=int)

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)

    matched_indices = linear_assignment(-iou_matrix)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Kalman:
    counter = 1

    def __init__(self, dets):
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
        self.kf.x[:4] = np.array([dets[0], dets[1], dets[2], dets[3]]).reshape((4, 1))
        self.id = Kalman.counter
        Kalman.counter += 1

    def __call__(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        return self.kf.x

    def correction(self, measurement):
        self.kf.update(measurement)

    def get_current_state(self):
        bbox = (np.array([self.kf.x[0], self.kf.x[1], self.kf.x[2], self.kf.x[3]]).reshape((1, 4)))
        return bbox


class FaceTracker:
    def __init__(self):
        self.current_tracker = []

    def __call__(self, detections):
        retain_trackers = []

        if len(self.current_tracker) == 0:
            for idx, det in enumerate(detections):
                tracker = Kalman(det)
                measurement = np.array([[int(det[0])], [int(det[1])], [int(det[2])], [int(det[3])]], np.float32)
                tracker.correction(measurement)
                self.current_tracker.append(tracker)

            for trk in self.current_tracker:
                d = trk.get_current_state()
                retain_trackers.append(np.concatenate((d[0], [trk.id])).reshape((1, -1)))

            if len(retain_trackers) > 0:
                return np.concatenate(retain_trackers)

            return np.empty((0, 5))

        else:
            predicted_trackers = []
            for t in range(len(self.current_tracker)):
                predictions = self.current_tracker[t]()[:4]
                predicted_trackers.append(predictions)

            predicted_trackers = np.asarray(predicted_trackers)

            matched, unmatched_detections, unmatched_trackers = associate_detections_to_trackers(detections[:, :-1],
                                                                                                 predicted_trackers)

            print('Matched Detections & Trackers', len(matched))
            print('Unmatched Detections', len(unmatched_detections))
            print('Unmatched Trackers', len(unmatched_trackers))
            print('Current Trackers', len(self.current_tracker))

            for t in range(len(self.current_tracker)):
                if (t not in unmatched_trackers):
                    d = matched[np.where(matched[:, 1] == t)[0], 0]
                    self.current_tracker[t].correction(np.array([detections[d, 0], detections[d, 1],
                                                                 detections[d, 2], detections[d, 3]]).reshape((4, 1)))

            for i in unmatched_detections:
                tracker = Kalman(detections[i, :-1])
                measurement = np.array([[int(detections[i, 0])], [int(detections[i, 1])], [int(detections[i, 2])],
                                        [int(detections[i, 3])]], np.float32)
                tracker.correction(measurement)
                self.current_tracker.append(tracker)

            for index in sorted(unmatched_trackers, reverse=True):
                del self.current_tracker[index]

            for trk in self.current_tracker:
                d = trk.get_current_x()
                retain_trackers.append(np.concatenate((d[0], [trk.id])).reshape(1, -1))

            if len(retain_trackers) > 0:
                return np.concatenate(retain_trackers)

            return np.empty((0, 5))


if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    cap = cv2.VideoCapture(0)

    f_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    f_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    detector = FaceDetector(sess=None)

    facetracker = FaceTracker()

    fps = FPS()
    ret, frame = cap.read()
    frame_number = 1

    fps.start()
    while ret:

        fps.update()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if cv2.waitKey(1) == ord("q"):
            break

        if (frame_number % 5 == 0) or (frame_number == 1):
            boxes = list()
            for _, box in detector.extract_faces(frame, f_w, f_h):
                boxes.append(box)

        trackers = facetracker(boxes)
        frame_number += 1

        print(trackers)

        cv2.imshow("Main", frame)

    cap.release()
    fps.stop()
    print(fps.fps())
