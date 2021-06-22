import imagezmq
import cv2


class Hub:
    def __init__(self):
        pass

    def run(self):
        image_hub = imagezmq.ImageHub()

        print(" Hub is starting ...")

        while True:
            msg, frame = image_hub.recv_image()
            cv2.imshow('Main', frame)
            print(msg)
            cv2.waitKey(1)
            image_hub.send_reply(b'Ok')
