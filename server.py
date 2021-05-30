from datetime import datetime
import numpy as np
import imagezmq
import argparse
import cv2

imageHub = imagezmq.ImageHub()

while True:
    (rpiName, frame) = imageHub.recv_image()
    imageHub.send_reply(b'OK')

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    cv2.imshow("main", frame)

cv2.destroyAllWindows()
