import gin
import tensorflow as tf
import random

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
def augment(image, label):
    operations = ['Rotation90', 'Rotation180', 'Rotation270', 'Flippinglr', 'Flippingud', 'Noise']
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
        elif operation == 'Noise':
            noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=1.0, dtype=tf.float32)
            image = tf.add(image, noise)
    return image, label