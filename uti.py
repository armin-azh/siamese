def conf_layer_gen(layer, cv1_out, cv1_filter,cv1_strides, cv2_out, cv2_filter, cv2_strides, padding):
    return {
        'layer': layer,
        'cv1_out': cv1_out,
        'cv1_filter': cv1_filter,
        'cv1_strides':cv1_strides,
        'cv2_out': cv2_out,
        'cv2_filter': cv2_filter,
        'cv2_strides': cv2_strides,
        'padding': padding
    }
