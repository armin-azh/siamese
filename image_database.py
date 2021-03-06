import os
from PIL import Image as Pil_Image
import numpy as np


class Image(object):
    """
    this class store Images
    """
    _default_size = (96, 96, 3)  # default size, referenced by Face Net article
    _default_format = None  # if None, every format will accepted

    def __init__(self, im_path):
        if Image.check_image_file(im_path):
            self._im_path = im_path
        else:
            raise ValueError

    @staticmethod
    def check_file_format(im_path):
        """ check file format
        :return bool
        """
        _, f_format = os.path.splitext(im_path)
        if Image.get_default_format() is None:
            return True
        if f_format[1:] in Image.get_default_format():
            return True
        else:
            return False

    @staticmethod
    def check_image_file(im_path):
        """
        check file existence and format
        :param im_path:
        :return: bool
        """
        if os.path.isfile(im_path) and Image.check_file_format(im_path):
            return True
        else:
            return False

    @classmethod
    def get_default_format(cls):
        return cls._default_format

    @classmethod
    def set_default_format(cls, format_list):
        """
        :param format_list: (list)
        :return: None
        """
        if isinstance(format_list, list) and len(format_list) > 0:
            cls._default_format = format_list
        else:
            cls._default_format = None

    @classmethod
    def get_default_size(cls):
        return cls._default_size

    @classmethod
    def set_default_size(cls, size):
        if len(size) == 3 and size[2] == 3:
            cls._default_size = size
        else:
            print("Set like this shape is not acceptable!")

    def read(self, numpy_format=False):
        """
        this method read image from path
        :return: numpy.array
        """
        im = Pil_Image.open(self._im_path)
        im = self.resize(im, save=True)
        if numpy_format:
            return np.array(im)
        else:
            return im

    @staticmethod
    def check_size(pil_image):
        """
        :param pil_image: image in PIL.Image class
        :return: bool
        """
        if pil_image.size == Image.get_default_size()[0:2]:
            return True
        else:
            return False

    def resize(self, pil_image, save=True):
        """
        this method check image size and then resize them to default size.
        :param save: whether save in it`s location
        :param pil_image: image in PIL.Image
        :return: Pil.Image
        """
        if Image.check_size(pil_image):
            return pil_image
        else:
            pil_image = pil_image.resize(Image.get_default_size()[0:2])
            if save:
                pil_image.save(self._im_path)
            return pil_image

    def encode(self, model):
        """
        get image in face net encoded
        :param model: keras model
        :return: tensor in shape (1,128)
        """
        im = self.read(numpy_format=True)
        im = im[..., ::-1]
        im = np.around(np.transpose(im, (2, 0, 1)) / 255., decimals=12)
        input_im = np.array([im])
        return model.predict(input_im)


class Identity(object):
    """
    Identity class
    """
    identities_name = []

    def __init__(self, name):
        if not Identity.check_identity_name(name):
            self._name = Identity.add_identity_name(name)
        else:
            raise ValueError(f'This name {name} is exists.')
        self._images = []

    @classmethod
    def add_identity_name(cls, name):
        cls.identities_name.append(name)
        return name

    @classmethod
    def delete_identity_name(cls, name):
        cls.identities_name.remove(name)

    def add_image(self, im_path):
        """
        add new image to the identity
        :param im_path:
        :return: None
        """
        self._images.append(Image(im_path=im_path))

    def get_images(self, numpy_format=False, n_images=None):
        """
        get list of images
        :param n_images: number of images that we want to fetch
        :param numpy_format: bool
        :return: list of images
        """
        images = list()
        last = self.number_images if (n_images is None or n_images >= self.number_images) else n_images
        for im in self._images[0:last]:
            im = im.read(numpy_format=numpy_format)
            images.append(im)

        return images

    def get_encoded_images(self, model, n_images=None):
        """
        get images in face net encode
        :return: list of tensor in shape(1x128)
        """
        en_images = list()
        last = self.number_images if (n_images is None or n_images >= self.number_images) else n_images
        for im in self._images[0:last]:
            en_images.append(im.encode(model))
        return en_images

    @property
    def number_images(self):
        """
        :return: int
        """
        return len(self._images)

    @classmethod
    def check_identity_name(cls, name):
        """
        check that the name is exists
        :param name:
        :return: bool
        """
        if name in cls.identities_name:
            return True
        else:
            return False

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, n_name):
        if not Identity.check_identity_name(n_name):
            Identity.delete_identity_name(self._name)
            self.add_identity_name(n_name)
        else:
            print(f'This name ->{n_name}<- is exist')

    def __del__(self):
        """
        delete identity from identity list
        """
        Identity.delete_identity_name(self._name)


class ImageDatabase(object):
    """
    virtual in memory database
    """

    def __init__(self, db_path):
        if os.path.isdir(db_path):
            self._db_path = db_path
        else:
            raise ValueError(f"The db location ->{db_path}<- is not exists.")
        self._identities = list()
        self._status = ''
        self.parse()

    def parse(self):
        """
        parse db_path database
        :return:
        """
        image_names = os.listdir(self._db_path)

        for name in image_names:
            ims_paths = os.path.join(self._db_path, name)
            new_identity = Identity(name=name)

            for im_name in os.listdir(ims_paths):
                im_path = os.path.join(ims_paths, im_name)
                new_identity.add_image(im_path=im_path)

            self.add_new_identity(identity=new_identity)

        self._status = 'commit'

    def add_new_identity(self, identity):
        """
        append identity to the list
        :param identity: Identity
        :return: None
        """
        self._identities.append(identity)

    def add_new_image(self, identity_name, im_path):
        pass

    def get_identity_images(self, numpy_format=False, n_images=None):
        """
        get identities images
        :param numpy_format: bool
        :param n_images: int
        :return: dictionary of identities
        """
        res = dict()
        for identity in self._identities:
            res[identity.name] = identity.get_images(numpy_format=numpy_format, n_images=n_images)

        return res

    def get_encoded_identities_images(self, model, n_images=None):
        """
        get encoded identities images
        :param model: keras model
        :param n_images: int
        :return: dictionary of identities
        """
        res = dict()
        for identity in self._identities:
            res[identity.name] = identity.get_encoded_images(model, n_images=n_images)
        return res


if __name__ == '__main__':
    path = './database'
    db = ImageDatabase(db_path=path)
    print(db.get_identity_images())


