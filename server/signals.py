import sys
import time
from tools.logger import Logger


def control_c_signal_handler(sig, f):
    """
    get control-c signal from os
    :param sig:
    :param f:
    :return:
    """
    l = Logger()
    l.warn("[Shutdown] Server is shutting down..")
    time.sleep(2)
    sys.exit(0)
