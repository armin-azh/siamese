import tensorflow.keras as K
import tensorflow as tf
import tensorflow.keras.backend as K_1
from tensorflow.keras.layers import (Conv2D,
                                     BatchNormalization,
                                     Activation,
                                     ZeroPadding2D,
                                     MaxPooling2D,
                                     concatenate,
                                     AveragePooling2D,
                                     Input,
                                     Flatten,
                                     Lambda,
                                     Dense
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
        num = '' if cv2_out is None else '1'

        self._main_layers = [
            Conv2D(cv1_out, cv1_filter, strides=cv1_strides, data_format=data_format, name=layer + '_conv' + num),
            BatchNormalization(axis=1, epsilon=epsilon, name=layer + '_bn' + num),
            Activation('relu'),
        ]
        if padding is not None:
            self._main_layers.append(ZeroPadding2D(padding=padding, data_format=data_format))
        if cv2_out is not None:
            self._main_layers.append(
                Conv2D(cv2_out, cv2_filter, strides=cv2_strides, data_format=data_format, name=layer + '_conv' + '2'))
            self._main_layers.append(BatchNormalization(axis=1, epsilon=epsilon, name=layer + '_bn' + '2'))
            self._main_layers.append(Activation('relu'))
        super(InceptionConvSubBlock, self).__init__()

    def set_weights(self, weights):
        params = self.weights

    def call(self, inputs, **kwargs):

        for layer in self._main_layers:
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

    def set_weights(self, weights):
        pass

    def call(self, inputs, **kwargs):
        """
        pass inputs through the network
        :param inputs:
        :param kwargs:
        :return:
        """
        conv3x3 = self._main_layers[0](inputs)
        conv5x5 = self._main_layers[1](inputs)
        pool = self._main_layers[2](inputs)
        pool = self._main_layers[3](pool)
        conv1x1 = self._main_layers[4](inputs)
        return concatenate([conv3x3, conv5x5, pool, conv1x1], axis=1)

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
                                  cv1_out=64,
                                  cv1_filter=(1, 1),
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
                                  cv2_filter=None,
                                  cv2_strides=None,
                                  padding=None)
        ]
        super(Inception1BBlock, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """
        pass inputs through the network
        :param inputs:
        :param kwargs:
        :return:
        """
        conv3x3 = self._main_layers[0](inputs)
        conv5x5 = self._main_layers[1](inputs)
        pool = self._main_layers[2](inputs)
        pool = self._main_layers[3](pool)
        conv1x1 = self._main_layers[4](inputs)
        return concatenate([conv3x3, conv5x5, pool, conv1x1], axis=1)

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
        conv3x3 = self._main_layers[0](inputs)
        conv5x5 = self._main_layers[1](inputs)
        pool = self._main_layers[2](inputs)
        pool = self._main_layers[3](pool)
        return concatenate([conv3x3, conv5x5, pool], axis=1)

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
                                  cv1_out=256,
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
        conv3x3 = self._main_layers[0](inputs)
        conv5x5 = self._main_layers[1](inputs)
        pool = self._main_layers[2](inputs)
        pool = self._main_layers[3](pool)
        conv1x1 = self._main_layers[4](inputs)
        return concatenate([conv3x3, conv5x5, pool, conv1x1], axis=1)

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
        conv3x3 = self._main_layers[0](inputs)
        conv5x5 = self._main_layers[1](inputs)
        pool = self._main_layers[2](inputs)
        pool = self._main_layers[3](pool)
        return concatenate([conv3x3, conv5x5, pool], axis=1)

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
            # 1x1
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
        conv3x3 = self._main_layers[0](inputs)
        pool = self._main_layers[1](inputs)
        pool = self._main_layers[2](pool)
        conv1x1 = self._main_layers[3](inputs)
        return concatenate([conv3x3, pool, conv1x1], axis=1)

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
        conv3x3 = self._main_layers[0](inputs)
        pool = self._main_layers[1](inputs)
        pool = self._main_layers[2](pool)
        conv1x1 = self._main_layers[3](inputs)
        return concatenate([conv3x3, pool, conv1x1], axis=1)

    def get_config(self):
        base_config = super().get_config()
        layers_configs = [layer.get_config() for layer in self._main_layers]

        for conf in layers_configs:
            for key, value in conf.items():
                base_config[key] = value

        return base_config


def face_net(input_shape):
    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3), data_format='channels_first')(X_input)

    X = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), name='conv1', data_format='channels_first')(X)
    X = BatchNormalization(axis=1, name='bn1', )(X)
    X = Activation('relu')(X)

    X = ZeroPadding2D(padding=(1, 1), data_format='channels_first')(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=2, data_format='channels_first')(X)

    X = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), name='conv2', data_format='channels_first')(X)
    X = BatchNormalization(axis=1, epsilon=1e-5, name='bn2')(X)
    X = Activation('relu')(X)

    X = ZeroPadding2D((1, 1), data_format='channels_first')(X)

    X = Conv2D(192, (3, 3), (1, 1), name='conv3', data_format='channels_first')(X)
    X = BatchNormalization(axis=1, epsilon=1e-5, name='bn3')(X)
    X = Activation('relu')(X)

    X = ZeroPadding2D((1, 1), data_format='channels_first')(X)
    X = MaxPooling2D(pool_size=3, strides=2, data_format='channels_first')(X)

    X = Inception1ABlock()(X)
    X = Inception1BBlock()(X)
    X = Inception1CBlock()(X)

    X = Inception2ABlock()(X)
    X = Inception2BBlock()(X)

    X = Inception3ABlock()(X)
    X = Inception3BBlock()(X)

    X = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), data_format='channels_first')(X)
    X = Flatten()(X)
    X = Dense(128, name='dense_layer')(X)

    X = Lambda(lambda x: K_1.l2_normalize(x, axis=1))(X)

    model = K.Model(inputs=X_input, outputs=X, name='FaceRecoModel')

    return model
