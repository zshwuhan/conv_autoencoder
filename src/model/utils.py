import scipy.misc
import numpy as np
import tensorflow as tf


def lrelu(x, leak=0.2):
    with tf.variable_scope("leaky_relu"):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def get_image(image_path, image_size, is_crop=True):
    img = transform(imread(image_path), image_size, is_crop)
    return img


def transform(image, npx=64, is_crop=True):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx)
    else:
        cropped_image = image
    return np.array(cropped_image) / 127.5 - 1.


def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    return scipy.misc.imresize(x[j:j + crop_h, i:i + crop_w],
                               [resize_w, resize_w])


def imread(path):
    return scipy.misc.imread(path).astype(np.float)


def normalize(image):
    return image / 2 + 0.5


def save_image(fig, folder_path, image_name, dpi=100):
    import os
    import matplotlib.pyplot as plt

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    fig.savefig(os.path.join(folder_path, image_name), dpi=dpi, bbox_inches="tight")
    print("Saved figure " + image_name + ".")
    plt.cla()


class ImageCorruption:
    @staticmethod
    def noise(x):
        return tf.mul(x, tf.cast(tf.random_uniform(shape=tf.shape(x),
                                                   minval=0,
                                                   maxval=2,
                                                   dtype=tf.int32), tf.float32))

    @staticmethod
    def box(x, box_size=20):
        x_shape = x.get_shape().as_list()

        rx = np.random.randint(0, x_shape[1] - box_size)
        ry = np.random.randint(0, x_shape[2] - box_size)
        # rx = tf.random_uniform(shape=1, maxval=x_shape[1] - box_size, dtype=tf.int32)
        # ry = tf.random_uniform(shape=1, maxval=x_shape[2] - box_size, dtype=tf.int32)

        # hack to implement masking because tf doesn't allow padding with constants other than zeros
        zeros = tf.zeros((box_size, box_size, 3))
        ones = tf.ones((box_size, box_size, 3))
        rez = zeros - ones
        mask_minus_one = tf.image.pad_to_bounding_box(rez, rx, ry, x_shape[1], x_shape[2])
        mask = mask_minus_one + tf.ones_like(mask_minus_one)

        # mask = np.ones((x_shape[1], x_shape[2]))
        # mask[rx:rx + box_size, ry: ry + box_size] = 0
        #
        # indices = np.where(mask == 0)
        # indices = np.array([(x, y) for x, y in zip(indices[0], indices[1])])
        # sparse = tf.SparseTensor(indices, np.zeros_like(indices[:, 0]), shape=[x_shape[1], x_shape[2]])
        # dense = tf.sparse_tensor_to_dense(sparse, default_value=1)
        # dense_rgb = tf.transpose(tf.pack([mas, dense, dense]), perm=[1, 2, 0])
        return tf.mul(x, tf.cast(mask, tf.float32))
