import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.layers import (Conv2D,
                                     BatchNormalization,
                                     Activation,
                                     ZeroPadding2D,
                                     MaxPooling2D,
                                     concatenate,
                                     AveragePooling2D
                                     )


class InceptionConvSubBlock(K.layers.Layer):
    def __init__(self, data_format, epsilon, layer=None,
                 cv1_out=None,
                 cv1_filter=(1, 1),
                 cv1_strides=(1, 1),
                 cv2_out=None,
                 cv2_filter=(3, 3),
                 cv2_strides=(1, 1),
                 padding=None, ):
        num = '' if cv2_out == None else '1'

        self._main_layers = [
            Conv2D(cv1_out, cv1_filter, strides=cv1_strides, data_format=data_format, name=layer + '_conv' + num),
            BatchNormalization(axis=1, epsilon=epsilon, name=layer + '_bn' + num),
            Activation('relu'),
        ]
        if padding != None:
            self._main_layers.append(ZeroPadding2D(padding=padding, data_format=data_format))
        if cv2_out != None:
            self._main_layers.append(
                Conv2D(cv2_out, cv2_filter, strides=cv2_strides, data_format=data_format, name=layer + '_conv' + '2'))
            self._main_layers.append(BatchNormalization(axis=1, epsilon=epsilon, name=layer + '_bn' + '2'))
            self._main_layers.append(Activation('relu'))
        super(InceptionConvSubBlock, self).__init__()

    def call(self, inputs, **kwargs):

        for layer in self._main_layer:
            inputs = layer(inputs)

        return inputs

    def get_config(self):
        base_config = super().get_config()
        layers_configs = [layer.get_config() for layer in self._main_layers]

        for conf in layers_configs:
            for key, value in conf.items():
                base_config[key] = value

        return base_config


class Inception1ABlock(K.layers.Layer):
    def __init__(self, **kwargs):
        self._epsilon = 1e-5
        self._data_format = 'channels_first'
        self._main_layers = [
            # 3x3
            InceptionConvSubBlock(epsilon=self._epsilon, data_format=self._data_format, layer='inception_3a_3x3',
                                  cv1_out=96,
                                  cv1_filter=(1, 1),
                                  cv1_strides=(1, 1),
                                  cv2_out=128,
                                  cv2_filter=(3, 3),
                                  cv2_strides=(1, 1),
                                  padding=(1, 1)),

            # 5x5
            InceptionConvSubBlock(epsilon=self._epsilon, data_format=self._data_format, layer='inception_3a_5x5',
                                  cv1_out=16,
                                  cv1_filter=(1, 1),
                                  cv1_strides=(1, 1),
                                  cv2_out=32,
                                  cv2_filter=(5, 5),
                                  cv2_strides=(1, 1),
                                  padding=(2, 2)),

            # pool
            MaxPooling2D(pool_size=3, strides=2, data_format=self._data_format),
            InceptionConvSubBlock(epsilon=self._epsilon, data_format=self._data_format, layer='inception_3a_pool',
                                  cv1_out=32,
                                  cv1_filter=(1, 1),
                                  cv1_strides=(1, 1),
                                  cv2_out=None,
                                  cv2_filter=None,
                                  cv2_strides=None,
                                  padding=((3, 4), (3, 4))),

            # 1x1
            InceptionConvSubBlock(epsilon=self._epsilon, data_format=self._data_format, layer='inception_3a_1x1',
                                  cv1_out=64,
                                  cv1_filter=(1, 1),
                                  cv1_strides=(1, 1),
                                  cv2_out=None,
                                  cv2_filter=None,
                                  cv2_strides=None,
                                  padding=None)
        ]
        super(Inception1ABlock, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """
        pass inputs through the network
        :param inputs:
        :param kwargs:
        :return:
        """
        concatinate_list = list()

        for layer in self._main_layer[:2]:
            inputs = layer(inputs)
            concatinate_list.append(inputs)

        for layer in self._main_layer[2:4]:
            inputs = layer(inputs)

        concatinate_list.append(inputs)

        inputs = self._main_layer[4](inputs)

        concatinate_list.append(inputs)

        return concatenate(concatinate_list, axis=1)

    def get_config(self):
        base_config = super().get_config()
        layers_configs = [layer.get_config() for layer in self._main_layers]

        for conf in layers_configs:
            for key, value in conf.items():
                base_config[key] = value

        return base_config


class Inception1BBlock(K.layers.Layer):
    def __init__(self, **kwargs):
        self._epsilon = 1e-5
        self._data_format = 'channels_first'

        self._main_layers = [
            # 3x3
            InceptionConvSubBlock(epsilon=self._epsilon, data_format=self._data_format, layer='inception_3b_3x3',
                                  cv1_out=96,
                                  cv1_filter=(1, 1),
                                  cv1_strides=(1, 1),
                                  cv2_out=128,
                                  cv2_filter=(3, 3),
                                  cv2_strides=(1, 1),
                                  padding=(1, 1)),

            # 5x5
            InceptionConvSubBlock(epsilon=self._epsilon, data_format=self._data_format, layer='inception_3b_5x5',
                                  cv1_out=32,
                                  cv1_filter=(1, 1),
                                  cv1_strides=(1, 1),
                                  cv2_out=64,
                                  cv2_filter=(5, 5),
                                  cv2_strides=(1, 1),
                                  padding=(2, 2)),

            # pool
            AveragePooling2D(pool_size=(3, 3), strides=(3, 3), data_format=self._data_format),
            InceptionConvSubBlock(data_format=self._data_format, epsilon=self._epsilon, layer='inception_3b_pool',
                                  cv1_out=64, cv1_filter=(1, 1),
                                  cv1_strides=(1, 1),
                                  cv2_filter=None,
                                  cv2_strides=None,
                                  cv2_out=None,
                                  padding=(4, 4)),

            # 1x1
            InceptionConvSubBlock(epsilon=self._epsilon, data_format=self._data_format, layer='inception_3b_1x1',
                                  cv1_out=64,
                                  cv1_filter=(1, 1),
                                  cv1_strides=(1, 1),
                                  cv2_out=None,
                                  cv2_filter=(5, 5),
                                  cv2_strides=(1, 1),
                                  padding=(2, 2))
        ]
        super(Inception1BBlock, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """
        pass inputs through the network
        :param inputs:
        :param kwargs:
        :return:
        """
        concatinate_list = list()

        for layer in self._main_layer[:2]:
            inputs = layer(inputs)
            concatinate_list.append(inputs)

        for layer in self._main_layer[2:4]:
            inputs = layer(inputs)

        concatinate_list.append(inputs)

        inputs = self._main_layer[4](inputs)
        concatinate_list.append(inputs)

        return concatenate(concatinate_list, axis=1)

    def get_config(self):
        base_config = super().get_config()
        layers_configs = [layer.get_config() for layer in self._main_layers]

        for conf in layers_configs:
            for key, value in conf.items():
                base_config[key] = value

        return base_config


class Inception1CBlock(K.layers.Layer):
    def __init__(self, **kwargs):
        self._epsilon = 1e-5
        self._data_format = 'channels_first'

        self._main_layers = [
            # 3x3
            InceptionConvSubBlock(epsilon=self._epsilon, data_format=self._data_format, layer='inception_3c_3x3',
                                  cv1_out=128,
                                  cv1_filter=(1, 1),
                                  cv1_strides=(1, 1),
                                  cv2_out=256,
                                  cv2_filter=(3, 3),
                                  cv2_strides=(2, 2),
                                  padding=(1, 1)),

            # 5x5
            InceptionConvSubBlock(epsilon=self._epsilon, data_format=self._data_format, layer='inception_3c_5x5',
                                  cv1_out=32,
                                  cv1_filter=(1, 1),
                                  cv1_strides=(1, 1),
                                  cv2_out=64,
                                  cv2_filter=(5, 5),
                                  cv2_strides=(2, 2),
                                  padding=(2, 2)),

            # pool
            MaxPooling2D(pool_size=3, strides=2, data_format=self._data_format),
            ZeroPadding2D(padding=((0, 1), (0, 1)), data_format=self._data_format)

        ]
        super(Inception1CBlock, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """
        pass inputs through the network
        :param inputs:
        :param kwargs:
        :return:
        """
        concatinate_list = list()

        for layer in self._main_layer[0:2]:
            inputs = layer(inputs)

        for layer in self._main_layer[2:]:
            inputs = layer(inputs)

        concatinate_list.append(inputs)

        return concatenate(concatinate_list, axis=1)

    def get_config(self):
        base_config = super().get_config()
        layers_configs = [layer.get_config() for layer in self._main_layers]

        for conf in layers_configs:
            for key, value in conf.items():
                base_config[key] = value

        return base_config


class Inception2ABlock(K.layers.Layer):
    def __init__(self, **kwargs):
        self._epsilon = 1e-5
        self._data_format = 'channels_first'
        self._main_layers = [
            # 3x3
            InceptionConvSubBlock(epsilon=self._epsilon, data_format=self._data_format, layer='inception_4a_3x3',
                                  cv1_out=96,
                                  cv1_filter=(1, 1),
                                  cv1_strides=(1, 1),
                                  cv2_out=192,
                                  cv2_filter=(3, 3),
                                  cv2_strides=(1, 1),
                                  padding=(1, 1)),

            # 5x5
            InceptionConvSubBlock(epsilon=self._epsilon, data_format=self._data_format, layer='inception_4a_5x5',
                                  cv1_out=32,
                                  cv1_filter=(1, 1),
                                  cv1_strides=(1, 1),
                                  cv2_out=64,
                                  cv2_filter=(5, 5),
                                  cv2_strides=(1, 1),
                                  padding=(2, 2)),

            # pool
            AveragePooling2D(pool_size=(3, 3), strides=(3, 3), data_format=self._data_format),
            InceptionConvSubBlock(epsilon=self._epsilon, data_format=self._data_format, layer='inception_4a_pool',
                                  cv1_filter=(1, 1),
                                  cv1_out=128,
                                  cv1_strides=(1, 1),
                                  cv2_out=None,
                                  cv2_filter=None,
                                  cv2_strides=None,
                                  padding=(2, 2)),

            # 1x1
            InceptionConvSubBlock(epsilon=self._epsilon, data_format=self._data_format, layer='inception_4a_1x1',
                                  cv1_out=128,
                                  cv1_filter=(1, 1),
                                  cv1_strides=(1, 1),
                                  cv2_out=None,
                                  cv2_filter=None,
                                  cv2_strides=None,
                                  padding=None)
        ]
        super(Inception2ABlock, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """
        pass inputs through the network
        :param inputs:
        :param kwargs:
        :return:
        """
        concatinate_list = list()

        for layer in self._main_layer[:2]:
            inputs = layer(inputs)
            concatinate_list.append(inputs)

        for layer in self._main_layer[2:4]:
            inputs = layer(inputs)

        concatinate_list.append(inputs)

        inputs = self._main_layer[4](inputs)

        concatinate_list.append(inputs)

        return concatenate(concatinate_list, axis=1)

    def get_config(self):
        base_config = super().get_config()
        layers_configs = [layer.get_config() for layer in self._main_layers]

        for conf in layers_configs:
            for key, value in conf.items():
                base_config[key] = value

        return base_config


class Inception2BBlock(K.layers.Layer):
    def __init__(self, **kwargs):
        self._epsilon = 1e-5
        self._data_format = 'channels_first'
        self._main_layers = [
            # 3x3
            InceptionConvSubBlock(epsilon=self._epsilon, data_format=self._data_format, layer='inception_4e_3x3',
                                  cv1_filter=(1, 1),
                                  cv1_out=160,
                                  cv1_strides=(1, 1),
                                  cv2_out=256,
                                  cv2_filter=(3, 3),
                                  cv2_strides=(2, 2),
                                  padding=(1, 1)),

            # 5x5
            InceptionConvSubBlock(epsilon=self._epsilon, data_format=self._data_format, layer='inception_4e_5x5',
                                  cv1_out=64,
                                  cv1_filter=(1, 1),
                                  cv1_strides=(1, 1),
                                  cv2_strides=(2, 2),
                                  cv2_filter=(5, 5),
                                  cv2_out=128,
                                  padding=(2, 2)),

            # pool
            MaxPooling2D(pool_size=3, strides=2, data_format=self._data_format),
            ZeroPadding2D(padding=((0, 1), (0, 1)), data_format=self._data_format)

        ]
        super(Inception2BBlock, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """
        pass inputs through the network
        :param inputs:
        :param kwargs:
        :return:
        """
        concatinate_list = list()

        for layer in self._main_layer[:2]:
            inputs = layer(inputs)
            concatinate_list.append(inputs)

        for layer in self._main_layer[2:]:
            inputs = layer(inputs)

        concatinate_list.append(inputs)

        return concatenate(concatinate_list, axis=1)

    def get_config(self):
        base_config = super().get_config()
        layers_configs = [layer.get_config() for layer in self._main_layers]

        for conf in layers_configs:
            for key, value in conf.items():
                base_config[key] = value

        return base_config


class Inception3ABlock(K.layers.Layer):
    def __init__(self, **kwargs):
        self._epsilon = 1e-5
        self._data_format = 'channels_first'
        self._main_layers = [
            # 3x3
            InceptionConvSubBlock(epsilon=self._epsilon, data_format=self._data_format, layer='inception_5a_3x3',
                                  cv1_out=96,
                                  cv1_filter=(1, 1),
                                  cv1_strides=(1, 1),
                                  cv2_out=384,
                                  cv2_filter=(3, 3),
                                  cv2_strides=(1, 1),
                                  padding=(1, 1)),

            # pool
            AveragePooling2D(pool_size=(3, 3), strides=(3, 3), data_format=self._data_format),
            InceptionConvSubBlock(epsilon=self._epsilon, data_format=self._data_format, layer='inception_5a_pool',
                                  cv1_out=96,
                                  cv1_filter=(1, 1),
                                  cv1_strides=(1, 1),
                                  cv2_out=None,
                                  cv2_filter=None,
                                  cv2_strides=None,
                                  padding=(1, 1)),
            InceptionConvSubBlock(epsilon=self._epsilon, data_format=self._data_format, layer='inception_5a_1x1',
                                  cv1_out=256,
                                  cv1_filter=(1, 1),
                                  cv1_strides=(1, 1),
                                  cv2_out=None,
                                  cv2_filter=None,
                                  cv2_strides=None,
                                  padding=None)

        ]
        super(Inception3ABlock, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """
        pass inputs through the network
        :param inputs:
        :param kwargs:
        :return:
        """
        concatinate_list = list()

        for layer in self._main_layer[:1]:
            inputs = layer(inputs)
            concatinate_list.append(inputs)

        for layer in self._main_layer[1:3]:
            inputs = layer(inputs)

        concatinate_list.append(inputs)

        inputs = self._main_layers[3]

        concatinate_list.append(inputs)

        return concatenate(concatinate_list, axis=1)

    def get_config(self):
        base_config = super().get_config()
        layers_configs = [layer.get_config() for layer in self._main_layers]

        for conf in layers_configs:
            for key, value in conf.items():
                base_config[key] = value

        return base_config


class Inception3BBlock(K.layers.Layer):
    def __init__(self, **kwargs):
        self._epsilon = 1e-5
        self._data_format = 'channels_first'
        self._main_layers = [
            # 3x3
            InceptionConvSubBlock(epsilon=self._epsilon, data_format=self._data_format, layer='inception_5b_3x3',
                                  cv1_out=96,
                                  cv1_filter=(1, 1),
                                  cv2_out=384,
                                  cv2_filter=(3, 3),
                                  cv2_strides=(1, 1),
                                  padding=(1, 1)),

            # pool
            MaxPooling2D(pool_size=3, strides=2, data_format=self._data_format),
            InceptionConvSubBlock(epsilon=self._epsilon, data_format=self._data_format, layer='inception_5b_pool',
                                  cv1_out=96,
                                  cv1_filter=(1, 1),
                                  cv1_strides=(1, 1),
                                  cv2_out=None,
                                  cv2_filter=None,
                                  cv2_strides=None,
                                  padding=(1, 1)),

            # 1x1
            InceptionConvSubBlock(epsilon=self._epsilon, data_format=self._data_format, layer='inception_5b_1x1',
                                  cv1_out=256,
                                  cv1_filter=(1, 1),
                                  cv1_strides=(1, 1),
                                  cv2_strides=None,
                                  cv2_filter=None,
                                  cv2_out=None,
                                  padding=None)

        ]
        super(Inception3BBlock, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """
        pass inputs through the network
        :param inputs:
        :param kwargs:
        :return:
        """
        concatinate_list = list()

        for layer in self._main_layer[:1]:
            inputs = layer(inputs)
            concatinate_list.append(inputs)

        for layer in self._main_layer[1:3]:
            inputs = layer(inputs)

        concatinate_list.append(inputs)

        inputs = self._main_layers[3]

        concatinate_list.append(inputs)

        return concatenate(concatinate_list, axis=1)

    def get_config(self):
        base_config = super().get_config()
        layers_configs = [layer.get_config() for layer in self._main_layers]

        for conf in layers_configs:
            for key, value in conf.items():
                base_config[key] = value

        return base_config

