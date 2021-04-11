import os
import configparser
from settings import BASE_DIR
import unittest
from component import Image
import utils


class DatabaseModule(unittest.TestCase):
    def setUp(self) -> None:
        conf = configparser.ConfigParser()
        conf.read(os.path.join(BASE_DIR, "conf.ini"))
        self.image_conf = conf['Image']

    def test_image_class_default_size(self):
        """
        test for read size from conf.ini file
        """
        size = Image.get_size()
        conf_size = (int(self.image_conf.get('width')), int(self.image_conf.get('height')))
        self.assertEqual(size[0], conf_size[0])
        self.assertEqual(size[1], conf_size[1])

    def test_image_file_path(self):
        filename = "/home/sephirod/xs_1.jpg"
        base_dir, file_path = utils.extract_filename(filename)
        json_path = os.path.join(base_dir, file_path + '.json')
        csv_path = os.path.join(base_dir, file_path + '.csv')
        im = Image(im_path=filename)
        self.assertEqual(im.image_path, filename)
        self.assertEqual(im.image_json_path, json_path)
        self.assertEqual(im.image_csv_path, csv_path)


if __name__ == "__main__":
    unittest.main()
