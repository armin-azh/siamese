from uti import conf_layer_gen

INCEPTION_LAYER_CONF = {
    'inception_a1': {
        '3x3': conf_layer_gen(layer='inception_3a_3x3',
                              cv1_out=96,
                              cv1_filter=(1, 1),
                              cv1_strides=(1, 1),
                              cv2_out=128,
                              cv2_filter=(3, 3),
                              cv2_strides=(1, 1),
                              padding=(1, 1)),
        '5x5': conf_layer_gen(layer='inception_3a_5x5',
                              cv1_out=16,
                              cv1_filter=(1, 1),
                              cv1_strides=(1, 1),
                              cv2_out=32,
                              cv2_filter=(5, 5),
                              cv2_strides=(1, 1),
                              padding=(2, 2)),
        'pool': {
            'pool_size': 3,
            'pool_stride': 2,
            'conv': {
                'filters': 32,
                'kernel': (1, 1),
                'name': 'inception_3a_pool_conv',
            },
            'batch_name': 'inception_3a_pool_bn',
            'zero_padd': ((3, 4), (3, 4))
        },
        '1x1': {
            'conv': {
                'filters': 64,
                'kernel': (1, 1),
                'name': 'inception_3a_1x1_conv'
            },
            'batch_name': 'inception_3a_1x1_bn'

        }

    },
    'inception_b1': {
        '3x3': conf_layer_gen(layer='inception_3b_3x3',
                              cv1_out=96,
                              cv1_filter=(1, 1),
                              cv1_strides=(1, 1),
                              cv2_out=128,
                              cv2_filter=(3, 3),
                              cv2_strides=(1, 1),
                              padding=(1, 1)),
        '5x5': conf_layer_gen(layer='inception_3b_5x5',
                              cv1_out=32,
                              cv1_filter=(1, 1),
                              cv1_strides=(1, 1),
                              cv2_out=64,
                              cv2_filter=(5, 5),
                              cv2_strides=(1, 1),
                              padding=(2, 2)),

        'pool': {
            'pool_size': (3, 3),
            'pool_stride': (3, 3),
            'conv': {
                'filters': 64,
                'kernel': (1, 1),
                'name': 'inception_3b_pool_conv',
            },
            'batch_name': 'inception_3b_pool_bn',
            'zero_padd': (4, 4)
        },
        '1x1': {
            'conv': {
                'filters': 64,
                'kernel': (1, 1),
                'name': 'inception_3b_1x1_conv'
            },
            'batch_name': 'inception_3b_1x1_bn'

        }
    },

}
