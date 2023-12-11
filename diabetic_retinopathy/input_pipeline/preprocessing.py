import gin
import tensorflow as tf

# must run imbalanced-learn==0.7.0 scikit-learn==0.23.2 numpy==1.23.4
from imblearn.over_sampling import RandomOverSampler


def resample(dataset):
    """Dataset resampling: check and output data imbalance, then resample dataset"""
    # todo: check data imbalance
    # todo: output imbalance situation and plot
    # todo: random resample data

    return dataset


@gin.configurable
def preprocess(image, label, img_height, img_width):
    """Dataset preprocessing: Normalizing and resizing"""

    # Normalize image: `uint8` -> `float32`.
    image = tf.cast(image, tf.float32) / 255.

    # Resize image with pad to avoid distortions
    image = tf.image.resize_with_pad(image, img_height, img_width)

    return image, label


@gin.configurable()
def augment(image, label, operation, central_frac):
    """Data augmentation"""
    if 'Rotation90' in operation:  # rotate 90 degree
        image = tf.image.rot90(image)

    elif 'Rotation180' in operation:  # rotate 180 degree
        image = tf.image.rot90(image, 2)

    elif 'Rotation270' in operation:  # rotate 270 degree
        image = tf.image.rot90(image, 3)

    elif 'Flippinglr' in operation:  # flip left and right
        image = tf.image.flip_left_right(image)

    elif 'Flippingud' in operation:  # flip up and down
        image = tf.image.flip_up_down(image)

    elif 'Cropping' in operation:  # crop the central region of the image
        image = tf.image.central_crop(image, central_frac)

    # elif 'Shearing' in operation:  # shear the image
    #     image = tf.image.shear(image)  # no such method in tf.image

    return image, label