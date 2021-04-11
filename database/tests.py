import os
import configparser
from settings import BASE_DIR
import unittest
from component import Image, Identity
import utils
from itertools import chain
import random


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

    def test_identity_name(self):
        random.seed(500)

        names = [
            'Armin Azhdehnia',
            'Alireza Bagherinia',
            'Reza Hadadi'
        ]

        images = [
            '/home/sephirod/gallery/xs_1.jpg',
            '/home/sephirod/gallery/xs_2.jpg',
            '/home/sephirod/gallery/xs_3.jpg',
            '/home/sephirod/gallery/xs_4.jpg',
            '/home/sephirod/gallery/xs_5.jpg'
        ]

        identities = list()
        for name in chain(names):
            identities.append(Identity.create(name))

        for iden in chain(identities):
            iden.add_image(random.choice(images))

        self.assertEqual(Identity.identities_name, names)

        self.assertEqual(Identity.create(names[0]), None)

        for iden in chain(identities):
            for im in iden.get_images_path():
                self.assertIn(im.image_path, images)

        del identities[0]

        self.assertNotEqual(Identity.identities_name, names)


if __name__ == "__main__":
    unittest.main()
