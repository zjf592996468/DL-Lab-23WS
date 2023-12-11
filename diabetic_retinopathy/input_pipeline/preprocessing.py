import gin
import tensorflow as tf
import random
import tensorflow_addons as tfa
def resample(datasets):
    class_0_ds = datasets.filter(lambda image, label: label == 0)
    class_1_ds = datasets.filter(lambda image, label: label == 1)
    weights = [0.5, 0.5]  # 重采样权重
    resampled_ds = tf.data.experimental.sample_from_datasets([class_0_ds, class_1_ds], weights)
    return resampled_ds


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


def augment(image, label):
    operations = ['Rotation90', 'Rotation180', 'Rotation270', 'Flippinglr', 'Flippingud', 'Cropping','Shearing']
    chosen_operations = random.sample(operations, k=random.randint(1, len(operations)))  # Randomly choose one or more operations

    for operation in chosen_operations:
        if operation == 'Rotation90':
            image = tf.image.rot90(image)
        elif operation == 'Rotation180':
            image = tf.image.rot90(image, 2)
        elif operation == 'Rotation270':
            image = tf.image.rot90(image, 3)
        elif operation == 'Flippinglr':
            image = tf.image.flip_left_right(image)
        elif operation == 'Flippingud':
            image = tf.image.flip_up_down(image)
        elif operation == 'Cropping':
            # Randomly crop and resize back to 256x256
            cropped_size = [tf.random.uniform([], minval=180, maxval=256, dtype=tf.int32) for _ in range(2)]
            image = tf.image.random_crop(image, size=[cropped_size[0], cropped_size[1], 3])
            image = tf.image.resize(image, [256, 256])
        elif operation == 'Shearing':
            # Shearing using affine transformation, keeping image size constant
            shear_x = random.uniform(-0.3, 0.3)  # Shear magnitude along x-axis
            shear_y = random.uniform(-0.3, 0.3)  # Shear magnitude along y-axis
            image = tfa.image.transform(image, [1.0, shear_x, 0.0, shear_y, 1.0, 0.0, 0.0, 0.0],
                                        interpolation='NEAREST')
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
