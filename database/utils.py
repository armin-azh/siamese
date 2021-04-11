from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import json


def write_json(data: dict, filename: str) -> None:
    """
    write data to jason file
    :param filename:
    :param data:
    :return: None
    """
    with open(filename, 'w') as output:
        json.dump(data, output)
