import imagezmq
import cv2


class Hub:
    def __init__(self):
        pass

    def run(self):
        image_hub = imagezmq.ImageHub(open_port="tcp://127.0.0.1:5556")
        # sender = imagezmq.ImageSender(connect_to="tcp://127.0.0.1:5556")

        print(" Hub is starting ...")

        while True:
            msg, frame = image_hub.recv_image()
            cv2.imshow('Main', frame)
            print(msg)
            cv2.waitKey(1)
            image_hub.send_reply(b'Ok')
            # sender.send_image("hub", frame)


a=Hub()


a.run()