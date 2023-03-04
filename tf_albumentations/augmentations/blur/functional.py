import tensorflow as tf
from tensorflow import Tensor

def blur(img: Tensor, ksize: int) -> Tensor:
    """
    img shape is (NHWC)
    """
    dtype = img.dtype
    if not ksize % 2:
        raise
    # if shape == (N, H, W, C), then pad (0, ksize)
    C = img.shape[-1]
    pad_num = int((ksize-1) / 2)
    print(pad_num)
    paddings = tf.constant([[0, 0], [pad_num, pad_num], [pad_num, pad_num], [0, 0]])
    img = tf.pad(img, paddings, mode="REFLECT")
    filters = tf.ones((ksize, ksize, 3, 1), dtype=dtype)
    # img = tf.nn.conv2d(img, filters, [1, 1, 1, 1], 'VALID')
    img = tf.nn.depthwise_conv2d(img, filters, [1, 1, 1, 1], 'VALID')
    weight = tf.constant(ksize*ksize*C, dtype=dtype)
    img /= weight
    # tf.nn.convolution
    return img
