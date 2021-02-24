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
    def __init__(self,data_format,epsilon,layer=None,
              cv1_out=None,
              cv1_filter=(1, 1),
              cv1_strides=(1, 1),
              cv2_out=None,
              cv2_filter=(3, 3),
              cv2_strides=(1, 1),
              padding=None,):
        num = '' if cv2_out == None else '1'

        self._main_layer = [
            Conv2D(cv1_out, cv1_filter, strides=cv1_strides, data_format=data_format, name=layer + '_conv' + num),
            BatchNormalization(axis=1, epsilon=epsilon, name=layer + '_bn' + num),
            Activation('relu'),
        ]
        if padding != None:
            self._main_layer.append(ZeroPadding2D(padding=padding, data_format=data_format))
        if cv2_out != None:
            self._main_layer.append(Conv2D(cv2_out, cv2_filter, strides=cv2_strides, data_format='channels_first', name=layer+'_conv'+'2'))
            self._main_layer.append(BatchNormalization(axis=1, epsilon=epsilon, name=layer+'_bn'+'2'))
            self._main_layer.append(Activation('relu'))
        super(InceptionConvSubBlock, self).__init__(**kwargs)

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


class InceptionPoolSubBlock(K.layers.Layer):
    def __init__(self, epsilon, data_format, **kwargs):
        self._main_layer = [
            MaxPooling2D(pool_size=kwargs.pool_size, strides=kwargs.pool_size, data_format=data_format),
            Conv2D(filters=kwargs.conv.filters, kernel_size=kwargs.conv.kernel, data_format=data_format,
                   name=kwargs.conv.name),
            BatchNormalization(axis=1, epsilon=epsilon, name=kwargs.batch_name),
            Activation('relu'),
            ZeroPadding2D(padding=kwargs.zero_padd, data_format=data_format)
        ]
        super(InceptionPoolSubBlock, self).__init__(**kwargs)

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
        self._data_format = 'channel_first'
        _3x3 = kwargs.get('3x3')

        _5x5 = kwargs.get('5x5')

        _pool = kwargs.get('pool')

        _1x1 = kwargs.get('1x1')

        self._main_layer = [
            # 3x3
            InceptionConvSubBlock(epsilon=self._epsilon, data_format=self._data_format, **_3x3),

            # 5x5
            InceptionConvSubBlock(epsilon=self._epsilon, data_format=self._data_format, **_5x5),

            # pool
            InceptionPoolSubBlock(epsilon=self._epsilon, data_format=self._data_format, **_pool),

            # 1x1
            InceptionConvSubBlock(epsilon=self._epsilon,data_format=self._data_format,**_1x1)
        ]
        super(InceptionA1, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """
        pass inputs through the network
        :param inputs:
        :param kwargs:
        :return:
        """
        concatinate_list = list()

        for layer in self._main_layer:
            inputs = layer(inputs)
            concatinate_list.append(inputs)

        return concatenate(concatinate_list,axis=1)

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
        self._data_format = 'channel_first'
        _3x3 = kwargs.get('3x3')

        _5x5 = kwargs.get('5x5')

        _pool = kwargs.get('pool')

        _1x1 = kwargs.get('1x1')

        self._main_layer = [
            # 3x3
            InceptionConvSubBlock(epsilon=self._epsilon, data_format=self._data_format, **_3x3),

            # 5x5
            InceptionConvSubBlock(epsilon=self._epsilon, data_format=self._data_format, **_5x5),

            # pool
            InceptionPoolSubBlock(epsilon=self._epsilon, data_format=self._data_format, **_pool),

            # 1x1
            InceptionSubBlock(epsilon=self._epsilon,data_format=self._data_format,**_1x1)
        ]
        super(InceptionA1, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """
        pass inputs through the network
        :param inputs:
        :param kwargs:
        :return:
        """
        concatinate_list = list()

        for layer in self._main_layer:
            inputs = layer(inputs)
            concatinate_list.append(inputs)

        return concatenate(concatinate_list,axis=1)

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
        self._data_format = 'channel_first'
        _3x3 = kwargs.get('3x3')

        _5x5 = kwargs.get('5x5')

        _pool = kwargs.get('pool')


        self._main_layer = [
            # 3x3
            InceptionConvSubBlock(epsilon=self._epsilon, data_format=self._data_format, **_3x3),

            # 5x5
            InceptionConvSubBlock(epsilon=self._epsilon, data_format=self._data_format, **_5x5),

            # pool
            MaxPooling2D(pool_size=_pool.pool_size,strides=_pool.pool_stride,data_format=self._data_format),
            ZeroPadding2D(padding=_pool.zero_padd,data_format=self._data_format)

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

        inputs = self._main_layer[2](inputs)
        inputs = self._main_layer[3](inputs)

        concatinate_list.append(inputs)

        return concatenate(concatinate_list,axis=1)

    def get_config(self):
        base_config = super().get_config()
        layers_configs = [layer.get_config() for layer in self._main_layers]

        for conf in layers_configs:
            for key, value in conf.items():
                base_config[key] = value

        return base_config


