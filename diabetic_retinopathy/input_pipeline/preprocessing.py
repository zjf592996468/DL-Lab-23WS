import gin
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


def resample(datasets):
    class_0_ds = datasets.filter(lambda image, label: label == 0)
    class_1_ds = datasets.filter(lambda image, label: label == 1)
    weights = [0.5, 0.5]  # 重采样权重
    resampled_ds = tf.data.experimental.sample_from_datasets([class_0_ds, class_1_ds], weights)
    return resampled_ds

@gin.configurable
def preprocess(image, label, img_height, img_width):
    """Dataset preprocessing: Normalizing and resizing"""
    # Normalize image: `uint8` -> `float32`.
    image = tf.cast(image, tf.float32) / 255.

    # Resize image with pad to avoid distortions
    image = tf.image.resize_with_pad(image, img_height, img_width)

    return image, label


@gin.configurable()
def augment(image, label, seed):
    """Data augmentation with a fixed seed for reproducibility"""

    # Randomly rotate the image by +- 0.125pi
    num_rotations = tf.random.uniform(shape=(), minval=0, maxval=4, dtype=tf.int32, seed=seed)
    image = tf.image.rot90(image, k=num_rotations)

    # 50% possibility up to down flipping
    image = tf.image.random_flip_up_down(image, seed=seed)

    # 50% possibility left to right flipping
    image = tf.image.random_flip_left_right(image, seed=seed)

    # Randomly crop the image from left and right sides and scale it to the original size
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    in_h = image.shape[0]
    in_w = image.shape[1]
    scaling = tf.random.uniform([2], 0.8, 1)
    x_scaling = scaling[0]
    y_scaling = scaling[1]
    out_h = tf.cast(in_h * y_scaling, dtype=tf.int32)
    out_w = tf.cast(in_w * x_scaling, dtype=tf.int32)
    seed = np.random.randint(2020)
    image = tf.image.random_crop(image, size=[out_h, out_w, 3], seed=seed)
    image = tf.image.resize(image, size=(in_h, in_w))

    # Random shearing
    x_shear = tf.random.uniform([1], minval=-0.1, maxval=0.1, dtype=tf.float32)[0]
    y_shear = tf.random.uniform([1], minval=-0.1, maxval=0.1, dtype=tf.float32)[0]
    image = tfa.image.transform(image, [1.0, x_shear, 0, y_shear, 1.0, 0.0, 0.0, 0.0])
    # Random brightness
    image = tf.image.random_brightness(image, max_delta=0.1, seed=seed)

    # Random saturation
    image = tf.image.random_saturation(image, lower=0.75, upper=1.25, seed=seed)

    # Random hue
    image = tf.image.random_hue(image, max_delta=0.01, seed=seed)

    # Random contrast
    image = tf.image.random_contrast(image, lower=0.75, upper=1.25, seed=seed)

    return image, label