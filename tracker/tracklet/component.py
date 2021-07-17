import numpy as np
from .data_asoociation import associate_detections_to_trackers
from .kalman import KalmanBoxTracker


class Sort:

    def __init__(self, max_age=1, min_hits=3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0

    def update(self, dets, img_size, addtional_attribute_list, predict_num):
        """
        Params:
          dets - a numpy array of detections in the format [[x,y,w,h,score],[x,y,w,h,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE:as in practical realtime MOT, the detector doesn't run on every single frame
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()  # kalman predict ,very fast ,<1ms
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        if dets != []:
            matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks)

            # update matched trackers with assigned detections
            for t, trk in enumerate(self.trackers):
                if t not in unmatched_trks:
                    d = matched[np.where(matched[:, 1] == t)[0], 0]
                    trk.update(dets[d, :][0])
                    trk.face_addtional_attribute.append(addtional_attribute_list[d[0]])

            # create and initialise new trackers for unmatched detections
            for i in unmatched_dets:
                trk = KalmanBoxTracker(dets[i, :])
                trk.face_addtional_attribute.append(addtional_attribute_list[i])
                self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if dets == []:
                trk.update([])
            d = trk.get_state()
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if trk.time_since_update >= self.max_age or trk.predict_num >= predict_num or d[2] < 0 or d[3] < 0 or d[0] > \
                    img_size[1] or d[1] > img_size[0]:
                if len(trk.face_addtional_attribute) >= 5:
                    pass
                    # utils.save_to_file(root_dic, trk)
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))


class TrackLet:
    def __init__(self, face_threshold: float, detect_interval: int):
        self._tracker = Sort()
        self._face_threshold = face_threshold
        self._detect_interval = detect_interval

    def _judge_side_face(self, facial_landmarks: np.ndarray):
        wide_dist = np.linalg.norm(facial_landmarks[0] - facial_landmarks[1])
        high_dist = np.linalg.norm(facial_landmarks[0] - facial_landmarks[3])
        dist_rate = high_dist / wide_dist

        # cal std
        vec_A = facial_landmarks[0] - facial_landmarks[2]
        vec_B = facial_landmarks[1] - facial_landmarks[2]
        vec_C = facial_landmarks[3] - facial_landmarks[2]
        vec_D = facial_landmarks[4] - facial_landmarks[2]
        dist_A = np.linalg.norm(vec_A)
        dist_B = np.linalg.norm(vec_B)
        dist_C = np.linalg.norm(vec_C)
        dist_D = np.linalg.norm(vec_D)

        # cal rate
        high_rate = dist_A / dist_C
        width_rate = dist_C / dist_D
        high_ratio_variance = np.fabs(high_rate - 1.1)  # smaller is better
        width_ratio_variance = np.fabs(width_rate - 1)

        return dist_rate, high_ratio_variance, width_ratio_variance

    def detect(self, faces: np.ndarray, frame: np.ndarray, points: np.ndarray, frame_size: tuple) -> np.ndarray:
        attribute_list = []

        face_list = []

        for i, item in enumerate(faces):
            score = round(faces[i, 4], 6)
            if score > self._face_threshold:
                det = np.squeeze(faces[i, 0:4])

                # face rectangle
                det[0] = np.maximum(det[0], 0)
                det[1] = np.maximum(det[1], 0)
                det[2] = np.minimum(det[2], frame_size[1])
                det[3] = np.minimum(det[3], frame_size[0])
                face_list.append(item)

                # face cropped
                bb = np.array(det, dtype=np.int32)

                # use 5 face landmarks  to judge the face is front or side
                squeeze_points = np.squeeze(points[:, i])
                tolist = squeeze_points.tolist()
                facial_landmarks = []

                for j in range(5):
                    item = [tolist[j], tolist[(j + 5)]]
                    facial_landmarks.append(item)

                cropped = frame[bb[1]:bb[3], bb[0]:bb[2], :].copy()

                dist_rate, high_ratio_variance, width_rate = self._judge_side_face(np.array(facial_landmarks))

                item_list = [cropped, score, dist_rate, high_ratio_variance, width_rate]
                attribute_list.append(item_list)

        final_faces = np.array(face_list)

        trackers = self._tracker.update(final_faces, frame_size, attribute_list, self._detect_interval)

        return trackers
