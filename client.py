

from stream.source import VideoStream
import imagezmq
import socket
import time
from settings import SOURCE_CONF

print(SOURCE_CONF.get('server_ip'))
sender = imagezmq.ImageSender(connect_to=SOURCE_CONF.get('server_ip'))
rpiName = socket.gethostname()
vs = VideoStream().start()
time.sleep(2.0)

while True:
    frame = vs.read()
    sender.send_image(rpiName, frame)
