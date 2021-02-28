import tensorflow.keras as K
import tensorflow.keras.backend as K_1
from tensorflow.keras.layers import (Input,
                                     ZeroPadding2D,
                                     BatchNormalization,
                                     Conv2D,
                                     Activation,
                                     MaxPooling2D,
                                     AveragePooling2D,
                                     Dense,
                                     Flatten,
                                     Lambda
                                     )

from layers import (Inception1ABlock,
                    Inception1BBlock,
                    Inception1CBlock,
                    Inception2ABlock,
                    Inception2BBlock,
                    Inception3ABlock,
                    Inception3BBlock)


class FaceNet(K.Model):
    def __init__(self, input_shape, **kwargs):
        self._epsilon = 1e-5
        self._main_layers = [
            Input(input_shape),

            ZeroPadding2D(padding=(3, 3)),

            Conv2D(filters=64, kernel_size=(7, 7), name='conv1'),
            BatchNormalization(axis=1, name='bn1'),
            Activation('relu'),

            ZeroPadding2D(padding=(1, 1)),
            MaxPooling2D(pool_size=(3, 3), strides=2),

            Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), name='conv2'),
            BatchNormalization(axis=1, epsilon=self._epsilon, name='bn2'),
            Activation('relu'),

            ZeroPadding2D((1, 1)),

            Conv2D(192, (3, 3), strides=(1, 1), name='conv3'),
            BatchNormalization(axis=1, epsilon=self._epsilon, name='bn3'),
            Activation('relu'),

            ZeroPadding2D((1, 1)),
            MaxPooling2D(pool_size=3, strides=2),

            Inception1ABlock(),
            Inception1BBlock(),
            Inception1CBlock(),

            Inception2ABlock(),
            Inception2BBlock(),

            Inception3ABlock(),
            Inception3BBlock(),

            AveragePooling2D(pool_size=(3, 3), strides=(1, 1), data_format='channels_first'),
            Flatten(),
            Dense(128, name='dense_layer')
        ]
        super(FaceNet, self).__init__()

    def call(self, inputs, training=None, mask=None):
        for layer in self._main_layers:
            inputs = layer(inputs)

        inputs = Lambda(lambda x: K_1.l2_normalize(x, axis=1))(inputs)

        return inputs


if __name__ == '__main__':
    model = FaceNet(input_shape=(3,96,96))
    print()