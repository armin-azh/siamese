import tensorflow.keras as K
import tensorflow as tf
import tensorflow.keras.backend as K_1
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
                                     Lambda
                                     )

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

    def _triplet_loss(self, y_true, y_pred, alpha=.2):
        anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
        positive_distance = tf.reduce_sum(tf.square(anchor, positive), axis=-1)
        negative_distance = tf.reduce_sum(tf.square(anchor, negative), axis=-1)
        basic_distance = tf.add(tf.subtract(positive_distance, negative_distance), alpha)
        loss = tf.reduce_sum(tf.maximum(basic_distance, 0.0))
        return loss

    def build(self):
        return self._build_model()

    def build_and_save(self, optimizer, metric, save_path):
        loader = Loader()
        model = self.build()
        model.compile(optimizer=optimizer, loss=self._triplet_loss, metrics=metric)
        loader.load_weights(model_obj=model)
        model.save(save_path)


