import tensorflow.keras as K
import tensorflow as tf
import tensorflow.keras.backend as K_1
from tensorflow.keras import Model
from tensorflow.keras.layers import (Input,
                                     ZeroPadding2D,
                                     BatchNormalization,
                                     Conv2D,
                                     Activation,
                                     MaxPooling2D,
                                     AveragePooling2D,
                                     concatenate,
                                     Dense,
                                     Flatten,
                                     Lambda,
                                     Concatenate,
                                     add,
                                     GlobalAveragePooling2D,
                                     Dropout
                                     )
from functools import partial

try:
    from .utils import Loader
except:
    from utils import Loader


class FaceNet(object):
    def __init__(self, input_shape, weights_path=None):
        self._input_shape = input_shape
        self._weights_path = weights_path
        self._epsilon = 1e-5
        self._data_format = 'channels_first'

    def _base_conv_block(self, input_tensor, layer=None,
                         cv1_out=None,
                         cv1_filter=(1, 1),
                         cv1_strides=(1, 1),
                         cv2_out=None,
                         cv2_filter=(3, 3),
                         cv2_strides=(1, 1),
                         padding=None):
        num = '' if cv2_out is None else '1'
        tensor = Conv2D(cv1_out, cv1_filter, strides=cv1_strides, data_format=self._data_format,
                        name=layer + '_conv' + num)(
            input_tensor)
        tensor = BatchNormalization(axis=1, epsilon=self._epsilon, name=layer + '_bn' + num)(tensor)
        tensor = Activation('relu')(tensor)
        if padding is not None:
            tensor = ZeroPadding2D(padding=padding, data_format=self._data_format)(tensor)
        if cv2_out is not None:
            tensor = Conv2D(cv2_out, cv2_filter, strides=cv2_strides, data_format=self._data_format,
                            name=layer + '_conv' + '2')(tensor)
            tensor = BatchNormalization(axis=1, epsilon=self._epsilon, name=layer + '_bn' + '2')(tensor)
            tensor = Activation('relu')(tensor)
        return tensor

    def _inception_1a_block(self, input_tensor):
        tensor3x3 = self._base_conv_block(input_tensor, layer='inception_3a_3x3',
                                          cv1_out=96,
                                          cv1_filter=(1, 1),
                                          cv1_strides=(1, 1),
                                          cv2_out=128,
                                          cv2_filter=(3, 3),
                                          cv2_strides=(1, 1),
                                          padding=(1, 1))
        tensor5x5 = self._base_conv_block(input_tensor, layer='inception_3a_5x5',
                                          cv1_out=16,
                                          cv1_filter=(1, 1),
                                          cv1_strides=(1, 1),
                                          cv2_out=32,
                                          cv2_filter=(5, 5),
                                          cv2_strides=(1, 1),
                                          padding=(2, 2))
        tensor_pool = MaxPooling2D(pool_size=3, strides=2, data_format=self._data_format)(input_tensor)
        tensor_pool = self._base_conv_block(tensor_pool, layer='inception_3a_pool',
                                            cv1_out=32,
                                            cv1_filter=(1, 1),
                                            cv1_strides=(1, 1),
                                            cv2_out=None,
                                            cv2_filter=None,
                                            cv2_strides=None,
                                            padding=((3, 4), (3, 4)))
        tensor1x1 = self._base_conv_block(input_tensor, layer='inception_3a_1x1',
                                          cv1_out=64,
                                          cv1_filter=(1, 1),
                                          cv1_strides=(1, 1),
                                          cv2_out=None,
                                          cv2_filter=None,
                                          cv2_strides=None,
                                          padding=None)

        return concatenate([tensor3x3, tensor5x5, tensor_pool, tensor1x1], axis=1)

    def _inception_1b_block(self, tensor_input):

        tensor3x3 = self._base_conv_block(tensor_input, layer='inception_3b_3x3',
                                          cv1_out=96,
                                          cv1_filter=(1, 1),
                                          cv1_strides=(1, 1),
                                          cv2_out=128,
                                          cv2_filter=(3, 3),
                                          cv2_strides=(1, 1),
                                          padding=(1, 1))
        tensor5x5 = self._base_conv_block(tensor_input, layer='inception_3b_5x5',
                                          cv1_out=32,
                                          cv1_filter=(1, 1),
                                          cv1_strides=(1, 1),
                                          cv2_out=64,
                                          cv2_filter=(5, 5),
                                          cv2_strides=(1, 1),
                                          padding=(2, 2))
        tensor_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3), data_format=self._data_format)(tensor_input)
        tensor_pool = self._base_conv_block(tensor_pool, layer='inception_3b_pool',
                                            cv1_out=64,
                                            cv1_filter=(1, 1),
                                            cv1_strides=(1, 1),
                                            cv2_filter=None,
                                            cv2_strides=None,
                                            cv2_out=None,
                                            padding=(4, 4))
        tensor1x1 = self._base_conv_block(tensor_input, layer='inception_3b_1x1',
                                          cv1_out=64,
                                          cv1_filter=(1, 1),
                                          cv1_strides=(1, 1),
                                          cv2_out=None,
                                          cv2_filter=None,
                                          cv2_strides=None,
                                          padding=None)

        return concatenate([tensor3x3, tensor5x5, tensor_pool, tensor1x1], axis=1)

    def _inception_1c_block(self, tensor_input):
        tensor3x3 = self._base_conv_block(tensor_input, layer='inception_3c_3x3',
                                          cv1_out=128,
                                          cv1_filter=(1, 1),
                                          cv1_strides=(1, 1),
                                          cv2_out=256,
                                          cv2_filter=(3, 3),
                                          cv2_strides=(2, 2),
                                          padding=(1, 1))
        tensor5x5 = self._base_conv_block(tensor_input, layer='inception_3c_5x5',
                                          cv1_out=32,
                                          cv1_filter=(1, 1),
                                          cv1_strides=(1, 1),
                                          cv2_out=64,
                                          cv2_filter=(5, 5),
                                          cv2_strides=(2, 2),
                                          padding=(2, 2))

        tensor_pool = MaxPooling2D(pool_size=3, strides=2, data_format=self._data_format)(tensor_input)
        tensor_pool = ZeroPadding2D(padding=((0, 1), (0, 1)), data_format=self._data_format)(tensor_pool)

        return concatenate([tensor3x3, tensor5x5, tensor_pool], axis=1)

    def _inception_2a_block(self, tensor_input):
        tensor3x3 = self._base_conv_block(tensor_input, layer='inception_4a_3x3',
                                          cv1_out=96,
                                          cv1_filter=(1, 1),
                                          cv1_strides=(1, 1),
                                          cv2_out=192,
                                          cv2_filter=(3, 3),
                                          cv2_strides=(1, 1),
                                          padding=(1, 1))
        tensor5x5 = self._base_conv_block(tensor_input, layer='inception_4a_5x5',
                                          cv1_out=32,
                                          cv1_filter=(1, 1),
                                          cv1_strides=(1, 1),
                                          cv2_out=64,
                                          cv2_filter=(5, 5),
                                          cv2_strides=(1, 1),
                                          padding=(2, 2))
        tensor_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3), data_format=self._data_format)(tensor_input)
        tensor_pool = self._base_conv_block(tensor_pool, layer='inception_4a_pool',
                                            cv1_filter=(1, 1),
                                            cv1_out=128,
                                            cv1_strides=(1, 1),
                                            cv2_out=None,
                                            cv2_filter=None,
                                            cv2_strides=None,
                                            padding=(2, 2))

        tensor1x1 = self._base_conv_block(tensor_input, layer='inception_4a_1x1',
                                          cv1_out=256,
                                          cv1_filter=(1, 1),
                                          cv1_strides=(1, 1),
                                          cv2_out=None,
                                          cv2_filter=None,
                                          cv2_strides=None,
                                          padding=None)

        return concatenate([tensor3x3, tensor5x5, tensor_pool, tensor1x1], axis=1)

    def _inception_2b_block(self, tensor_input):
        tensor3x3 = self._base_conv_block(tensor_input, layer='inception_4e_3x3',
                                          cv1_filter=(1, 1),
                                          cv1_out=160,
                                          cv1_strides=(1, 1),
                                          cv2_out=256,
                                          cv2_filter=(3, 3),
                                          cv2_strides=(2, 2),
                                          padding=(1, 1))

        tensor5x5 = self._base_conv_block(tensor_input, layer='inception_4e_5x5',
                                          cv1_out=64,
                                          cv1_filter=(1, 1),
                                          cv1_strides=(1, 1),
                                          cv2_strides=(2, 2),
                                          cv2_filter=(5, 5),
                                          cv2_out=128,
                                          padding=(2, 2))

        tensor_pool = MaxPooling2D(pool_size=3, strides=2, data_format=self._data_format)(tensor_input)
        tensor_pool = ZeroPadding2D(padding=((0, 1), (0, 1)), data_format=self._data_format)(tensor_pool)

        return concatenate([tensor3x3, tensor5x5, tensor_pool], axis=1)

    def _inception_3a_block(self, tensor_input):
        tensor3x3 = self._base_conv_block(tensor_input, layer='inception_5a_3x3',
                                          cv1_out=96,
                                          cv1_filter=(1, 1),
                                          cv1_strides=(1, 1),
                                          cv2_out=384,
                                          cv2_filter=(3, 3),
                                          cv2_strides=(1, 1),
                                          padding=(1, 1))
        tensor_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3), data_format=self._data_format)(tensor_input)
        tensor_pool = self._base_conv_block(tensor_pool, layer='inception_5a_pool',
                                            cv1_out=96,
                                            cv1_filter=(1, 1),
                                            cv1_strides=(1, 1),
                                            cv2_out=None,
                                            cv2_filter=None,
                                            cv2_strides=None,
                                            padding=(1, 1))
        tensor1x1 = self._base_conv_block(tensor_input, layer='inception_5a_1x1',
                                          cv1_out=256,
                                          cv1_filter=(1, 1),
                                          cv1_strides=(1, 1),
                                          cv2_out=None,
                                          cv2_filter=None,
                                          cv2_strides=None,
                                          padding=None)

        return concatenate([tensor3x3, tensor_pool, tensor1x1], axis=1)

    def _inception_3b_block(self, tensor_input):
        tensor3x3 = self._base_conv_block(tensor_input, layer='inception_5b_3x3',
                                          cv1_out=96,
                                          cv1_filter=(1, 1),
                                          cv2_out=384,
                                          cv2_filter=(3, 3),
                                          cv2_strides=(1, 1),
                                          padding=(1, 1))
        tensor_pool = MaxPooling2D(pool_size=3, strides=2, data_format=self._data_format)(tensor_input)
        tensor_pool = self._base_conv_block(tensor_pool, layer='inception_5b_pool',
                                            cv1_out=96,
                                            cv1_filter=(1, 1),
                                            cv1_strides=(1, 1),
                                            cv2_out=None,
                                            cv2_filter=None,
                                            cv2_strides=None,
                                            padding=(1, 1))
        tensor1x1 = self._base_conv_block(tensor_input, layer='inception_5b_1x1',
                                          cv1_out=256,
                                          cv1_filter=(1, 1),
                                          cv1_strides=(1, 1),
                                          cv2_strides=None,
                                          cv2_filter=None,
                                          cv2_out=None,
                                          padding=None)
        return concatenate([tensor3x3, tensor_pool, tensor1x1], axis=1)

    def _build_model(self):
        X_input = Input(self._input_shape)

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

        X = self._inception_1a_block(X)
        X = self._inception_1b_block(X)
        X = self._inception_1c_block(X)

        X = self._inception_2a_block(X)
        X = self._inception_2b_block(X)

        X = self._inception_3a_block(X)
        X = self._inception_3b_block(X)

        X = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), data_format='channels_first')(X)

        X = Flatten()(X)
        X = Dense(128, name='dense_layer')(X)

        X = Lambda(lambda x: K_1.l2_normalize(x, axis=1))(X)

        return K.Model(inputs=X_input, outputs=X, name='FaceRecoModel')

    @staticmethod
    def triplet_loss(y_true, y_pred, alpha=.2):
        anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
        positive_distance = tf.reduce_sum(tf.square(anchor, positive), axis=-1)
        negative_distance = tf.reduce_sum(tf.square(anchor, negative), axis=-1)
        basic_distance = tf.add(tf.subtract(positive_distance, negative_distance), alpha)
        loss = tf.reduce_sum(tf.maximum(basic_distance, 0.0))
        return loss

    def build(self):
        return self._build_model()

    def build_and_save(self, optimizer, metric, save_path):
        loader = Loader(weight_path=self._weights_path)
        model = self.build()
        model.compile(optimizer=optimizer, loss=FaceNet.triplet_loss, metrics=metric)
        loader.load_weights(model_obj=model)
        model.save(save_path)


class InceptionResNetV1(object):
    def __init__(self, input_shape=(160, 160, 3), embed_size=128, bn_momentum=.995, epsilon=1e-3):
        """
        constructor
        :param input_shape: input tensor shape
        :param embed_size: output embedded tensor
        :param bn_momentum: batch normalization momentum argument
        :param epsilon:
        """
        self._model_name = 'inception_resnet_v1'
        self._embed_size = embed_size
        self._input_shape = input_shape
        self._epsilon = epsilon
        self._bn_momentum = bn_momentum

    @staticmethod
    def scaling_tensor(input_x, scale):
        """
        scale the input tensor
        :param input_x: tensor
        :param scale: float number
        :return: tensor
        """
        return input_x * scale

    @staticmethod
    def generate_name(name, branch_idx=None, prefix=None, ):
        """
        concatenate arguments to generate new name for specific layer
        :param name: (string) name than annotate the layer
        :param prefix: (string) block name
        :param branch_idx: (int) branch index in inception block
        :return: (string)
        """
        if prefix is None:
            return None
        if branch_idx is None:
            return '_'.join((prefix, name))
        else:
            return '_'.join((prefix, 'Branch', str(branch_idx), name))

    def _base_conv(self, input_x, filters, kernel_size, strides=1, padding='SAME', use_bias=False, block_name=None,
                   activation='relu'):
        """
        base convolution_batch norm block
        :param input_x: tensor
        :param filters: number of filters
        :param kernel_size: convolution kernel size
        :param strides: convolution stride
        :param padding: convolution padding
        :param use_bias: convolution bias usage
        :param block_name:
        :param activation:
        :return: tensor
        """
        input_x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, name=block_name,
                         use_bias=use_bias)(
            input_x)

        if not use_bias:
            bn_axis = 1 if K_1.image_data_format() == 'channels_first' else 3
            bn_name = self.generate_name(name='BatchNorm', prefix=block_name)
            input_x = BatchNormalization(axis=bn_axis, name=bn_name, scale=False, momentum=self._bn_momentum,
                                         epsilon=self._epsilon)(input_x)
        if activation is not None:
            activation_name = self.generate_name(name="Activation", prefix=block_name)
            input_x = Activation(activation=activation, name=activation_name)(input_x)
        return input_x

    def _tail_block(self, input_x, con_tensors, name_fmt_fn, activation, scale, channels_type):
        """
        this block is perform on each inception_resnet_block_xx
        :param input_x: input tensor
        :param con_tensors: concatenated tensors
        :param name_fmt_fn: generate name function
        :param activation:
        :param scale: (float)
        :param channels_type: channels axis
        :return: tensor
        """
        ls_tensor = self._base_conv(input_x=con_tensors, filters=K_1.int_shape(input_x)[channels_type], kernel_size=1,
                                    activation=None, use_bias=True, block_name=name_fmt_fn(name='Conv2d_1x1'))

        ls_tensor = Lambda(self.scaling_tensor, output_shape=K_1.int_shape(ls_tensor)[1:], arguments={'scale': scale})(
            ls_tensor)

        input_x = add([input_x, ls_tensor])

        if activation is not None:
            input_x = Activation(activation=activation, name=name_fmt_fn(name='Activation'))(input_x)

        return input_x

    def _inception_resnet_block_35(self, input_x, scale, block_idx, activation='relu'):
        """
        inception resnet block 35
        :param input_x: input tensor
        :param scale: scale
        :param block_idx: block index
        :param activation:
        :return: tensor
        """
        channels_type = 1 if K_1.image_data_format() == 'channels_first' else 3
        if block_idx is None:
            prefix = None
        else:
            prefix = 'Block35_' + str(block_idx)

        abs_name = partial(self.generate_name, prefix=prefix)

        tensor_1x1_0 = self._base_conv(input_x=input_x, filters=32, kernel_size=1,
                                       block_name=abs_name('Conv2d_1x1', 0))
        tensor_1x1_1 = self._base_conv(input_x=input_x, filters=32, kernel_size=1,
                                       block_name=abs_name(name='Conv2d_0a_1x1', branch_idx=1))
        tensor_3x3_1 = self._base_conv(input_x=tensor_1x1_1, filters=32, kernel_size=3,
                                       block_name=abs_name(name='Conv2d_0b_3x3', branch_idx=1))
        tensor_1x1_2 = self._base_conv(input_x=input_x, filters=32, kernel_size=1,
                                       block_name=abs_name(name='Conv2d_0a_1x1', branch_idx=2))
        tensor_3x3_2 = self._base_conv(input_x=tensor_1x1_2, filters=32, kernel_size=3,
                                       block_name=abs_name(name='Conv2d_0b_3x3', branch_idx=2))
        tensor_3x3_2 = self._base_conv(input_x=tensor_3x3_2, filters=32, kernel_size=3,
                                       block_name=abs_name(name='Conv2d_0c_3x3', branch_idx=2))
        tensors = [tensor_1x1_0, tensor_3x3_1, tensor_3x3_2]

        con_tensors = Concatenate(axis=channels_type, name=abs_name(name='Concatenate'))(tensors)

        return self._tail_block(input_x=input_x, con_tensors=con_tensors, name_fmt_fn=abs_name, activation=activation,
                                scale=scale, channels_type=channels_type)

    def _inception_res_net_block_17(self, input_x, scale, block_idx, activation='relu'):
        """
        inception res net block 17
        :param input_x:
        :param scale:
        :param block_idx:
        :param activation:
        :return: tensor
        """
        channels_type = 1 if K_1.image_data_format() == 'channels_first' else 3
        if block_idx is None:
            prefix = None
        else:
            prefix = 'Block17_' + str(block_idx)

        abs_name = partial(self.generate_name, prefix=prefix)

        tensor_1x1_0 = self._base_conv(input_x=input_x, filters=128, kernel_size=1,
                                       block_name=abs_name('Conv2d_1x1', 0))
        tensor_1x1_1 = self._base_conv(input_x=input_x, filters=128, kernel_size=1,
                                       block_name=abs_name('Conv2d_0a_1x1', 1))
        tensor_1x7_1 = self._base_conv(input_x=tensor_1x1_1, filters=128, kernel_size=[1, 7],
                                       block_name=abs_name('Conv2d_0b_1x7', 1))
        tensor_7x1_1 = self._base_conv(input_x=tensor_1x7_1, filters=128, kernel_size=[7, 1],
                                       block_name=abs_name('Conv2d_0c_7x1', 1))

        tensors = [tensor_1x1_0, tensor_7x1_1]
        con_tensors = Concatenate(axis=channels_type, name=abs_name(name='Concatenate'))(tensors)

        return self._tail_block(input_x=input_x, con_tensors=con_tensors, name_fmt_fn=abs_name, activation=activation,
                                scale=scale, channels_type=channels_type)

    def _inception_res_net_block_8(self, input_x, scale, block_idx, activation='relu'):
        """
        inception tes block 8
        :param input_x: tensor
        :param scale: scale
        :param block_idx:
        :param activation:
        :return: tensor
        """

        channels_type = 1 if K_1.image_data_format() == 'channels_first' else 3
        if block_idx is None:
            prefix = None
        else:
            prefix = 'Block8_' + str(block_idx)

        abs_name = partial(self.generate_name, prefix=prefix)

        tensor_1x1_0 = self._base_conv(input_x=input_x, filters=192, kernel_size=1,
                                       block_name=abs_name('Conv2d_1x1', 0))
        tensor_1x1_1 = self._base_conv(input_x=input_x, filters=192, kernel_size=1,
                                       block_name=abs_name('Conv2d_0a_1x1', 1))
        tensor_1x3_1 = self._base_conv(input_x=tensor_1x1_1, filters=192, kernel_size=[1, 3],
                                       block_name=abs_name('Conv2d_0b_1x3', 1))
        tensor_3x1_1 = self._base_conv(input_x=tensor_1x3_1, filters=192, kernel_size=[3, 1],
                                       block_name=abs_name('Conv2d_0c_3x1', 1))

        tensors = [tensor_1x1_0, tensor_3x1_1]

        con_tensors = Concatenate(axis=channels_type, name=abs_name(name='Concatenate'))(tensors)

        return self._tail_block(input_x=input_x, con_tensors=con_tensors, name_fmt_fn=abs_name, activation=activation,
                                scale=scale, channels_type=channels_type)

    def build(self, weight_path=None, dropout_keep_prob=.8, input_shape=None):
        if input_shape is None:
            input_tensor = Input(shape=self._input_shape)
        else:
            input_tensor = Input(shape=input_shape)

        tensor = self._base_conv(input_x=input_tensor, filters=32, kernel_size=3, strides=2, padding='VALID',
                                 block_name='Conv2d_1a_3x3')
        tensor = self._base_conv(input_x=tensor, filters=32, kernel_size=3, strides=2, padding='VALID',
                                 block_name='Conv2d_2a_3x3')
        tensor = self._base_conv(input_x=tensor, filters=64, kernel_size=3, block_name='Conv2d_2b_3x3')
        tensor = MaxPooling2D(pool_size=3, strides=2, name='MaxPool_3a_3x3')(tensor)
        tensor = self._base_conv(input_x=tensor, filters=80, kernel_size=1, padding='VALID', block_name='Conv2d_3b_1x1')
        tensor = self._base_conv(input_x=tensor, filters=192, kernel_size=3, padding='VALID',
                                 block_name='Conv2d_4a_3x3')
        tensor = self._base_conv(input_x=tensor, filters=256, kernel_size=3, strides=2, padding='VALID',
                                 block_name='Conv2d_4b_3x3')

        # 5 x inception block 35
        for idx in range(1, 6):
            tensor = self._inception_resnet_block_35(input_x=tensor, scale=.17, block_idx=idx)

        channel_type = 1 if K_1.image_data_format() == 'channels_first' else 3
        abs_name = partial(self.generate_name, prefix='Mixed_6a')

        tensor_3x3_0 = self._base_conv(input_x=tensor, filters=384, kernel_size=3, strides=2, padding='VALID',
                                       block_name=abs_name('Conv2d_1a_3x3', 0))
        tensor_1x1_1 = self._base_conv(input_x=tensor, filters=192, kernel_size=1, strides=1,
                                       block_name=abs_name('Conv2d_0a_1x1', 1))
        tensor_3x3_1 = self._base_conv(input_x=tensor_1x1_1, filters=192, kernel_size=3,
                                       block_name=abs_name('Conv2d_0b_3x3', 1))

        tensor_3x3_1 = self._base_conv(input_x=tensor_3x3_1, filters=256, kernel_size=3, strides=2, padding='VALID',
                                       block_name=abs_name('Conv2d_1a_3x3', 1))

        tensor_pool = MaxPooling2D(pool_size=3, strides=2, padding='VALID',
                                   name=abs_name('MaxPool_1a_3x3', 2))(tensor)
        tensors = [tensor_3x3_0, tensor_3x3_1, tensor_pool]
        tensor = Concatenate(axis=channel_type, name='Mixed_6a')(tensors)

        for idx in range(1, 11):
            tensor = self._inception_res_net_block_17(input_x=tensor, scale=.1, block_idx=idx)

        abs_name = partial(self.generate_name, prefix='Mixed_7a')
        tensor_1x1_0 = self._base_conv(input_x=tensor, filters=256, kernel_size=1,
                                       block_name=abs_name('Conv2d_0a_1x1', 0))
        tensor_3x3_0 = self._base_conv(input_x=tensor_1x1_0, filters=384, kernel_size=3, strides=2, padding='VALID',
                                       block_name=abs_name('Conv2d_1a_3x3', 0))
        tensor_1x1_1 = self._base_conv(input_x=tensor, filters=256, kernel_size=1,
                                       block_name=abs_name('Conv2d_0a_1x1', 1))

        tensor_3x3_1 = self._base_conv(input_x=tensor_1x1_1, filters=256, kernel_size=3, strides=2, padding='VALID',
                                       block_name=abs_name('Conv2d_1a_3x3', 1))

        tensor_1x1_2 = self._base_conv(input_x=tensor, filters=256, kernel_size=1,
                                       block_name=abs_name('Conv2d_0a_1x1', 2))

        tensor_3x3_2 = self._base_conv(input_x=tensor_1x1_2, filters=256, kernel_size=3,
                                       block_name=abs_name('Conv2d_0b_3x3', 2))

        tensor_3x3_2 = self._base_conv(input_x=tensor_3x3_2, filters=256, kernel_size=3, strides=2, padding='VALID',
                                       block_name=abs_name('Conv2d_1a_3x3', 2))

        tensor_pool = MaxPooling2D(pool_size=3, strides=2, padding='VALID',
                                   name=abs_name('MaxPool_1a_3x3', 3))(tensor)

        tensors = [tensor_3x3_0, tensor_3x3_1, tensor_3x3_2, tensor_pool]

        tensor = Concatenate(axis=channel_type, name='Mixed_7a')(tensors)

        for idx in range(1, 6):
            tensor = self._inception_res_net_block_8(input_x=tensor, scale=.2, block_idx=idx)

        tensor = self._inception_res_net_block_8(input_x=tensor, scale=1., activation=None, block_idx=6)

        tensor = GlobalAveragePooling2D(name='AvgPool')(tensor)

        tensor = Dropout(1. - dropout_keep_prob, name='Dropout')(tensor)

        tensor = Dense(self._embed_size, use_bias=False, name='Bottleneck')(tensor)

        batch_name = self.generate_name(name='BatchNorm', prefix='Bottleneck')
        tensor = BatchNormalization(momentum=self._bn_momentum, epsilon=self._epsilon, scale=False, name=batch_name)(
            tensor)

        model = Model(input_tensor, tensor, name=self._model_name)

        if weight_path is not None:
            model.load_weights(weight_path)

        return model


if __name__ == '__main__':
    model = InceptionResNetV1().build(weight_path='./saved_models/facenet_keras_weights.h5')
    model.summary()
    model.save('./saved_models/facenet_model_3.8.h5')
