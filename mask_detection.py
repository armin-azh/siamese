from mtcnn import MTCNN
import numpy as np
import cv2
from recognition.preprocessing import cvt_to_gray


def cut_mask(face, img):
    keypoints = face[0]['keypoints']
    box = face[0]['box']
    eye = keypoints['right_eye']
    x, y, w, h = box
    mask_height = y + h - eye[1]
    mask_width = x + w
    mask = img[eye[1] + 10:eye[1] + mask_height, x: mask_width, :]
    return mask


def mask_edging(mask):
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    mask_edge = cv2.Laplacian(gray_mask, cv2.CV_8U)
    return mask_edge


def mask_detector(face, img):
    keypoints = face[0]['keypoints']
    nose = keypoints["nose"]
    nose = img[nose[1] - 10: nose[1] + 10, nose[0] - 10: nose[0] + 10, :]
    mouth_l = keypoints["mouth_left"]
    mouth_r = keypoints["mouth_right"]
    mouth = img[mouth_r[1] - 10: mouth_r[1] + 10, mouth_l[0]:mouth_r[0], :]
    mouth = cv2.cvtColor(mouth, cv2.COLOR_RGB2GRAY)
    nose = cv2.cvtColor(nose, cv2.COLOR_RGB2GRAY)
    # not sure how to compare efficiently yet
    av_mouth = np.average(mouth)
    av_nose = np.average(nose)
    av_dist = abs(av_nose - av_mouth)
    th = 10
    if av_dist < th:
        return True
    else:
        return False


def put_mask(img, face, mask):
    keypoints = face[0]['keypoints']
    box = face[0]['box']
    eye = keypoints['right_eye']
    x, y, w, h = box
    mask_height = y + h - eye[1]
    mask_width = x + w
    img[eye[1] + 10:eye[1] + mask_height, x: mask_width, :] = mask
    return img

# im = cv2.imread("/content/sample_data/images.jpg")
# im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
# face_detector = MTCNN()
# face = face_detector.detect_faces(im)


def mask_detection(img, face):
    hasmask = mask_detector(img, face)
    if hasmask:
        mask = cut_mask(img, face)
        mask_edge = mask_edging(mask)
        mask_edge = cvt_to_gray(mask_edge)
        img_mask = put_mask(img, face, mask_edge)
        return img_mask
    else:
        return img



