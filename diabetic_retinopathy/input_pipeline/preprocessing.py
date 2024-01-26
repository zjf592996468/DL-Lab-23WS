import gin
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import numpy as np


def check_imb(labels):
    """check and plot imbalance situation, return num of classes and num of samples in each class"""
    # Calculate the num of each class
    label_counts = labels["Retinopathy grade"].value_counts().sort_index()

    # Set different color for each class
    colors = plt.cm.tab10(range(label_counts.index.shape[0]))  # Options: Accent, tab10, Paired

    # Plot the figure
    plt.figure(figsize=(8, 6))
    bars = plt.bar(label_counts.index, label_counts.values, color=colors, label='Class percentages')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution')
    plt.xticks(label_counts.index)

    # Show num on the bar
    for i, count in enumerate(label_counts.values):
        plt.text(label_counts.index[i], count, str(count), ha='center', va='bottom')

    # Show the legend with percent info
    total_samples = sum(label_counts.values)
    percents = [count / total_samples * 100 for count in label_counts.values]
    legend_labels = [f'Class {label_counts.index[i]}: {percents[i]:.2f}%' for i in range(label_counts.index.shape[0])]
    plt.legend(bars, legend_labels, bbox_to_anchor=(1, 1))
    plt.tight_layout()

    return plt


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
    # Randomly rotate the image by +- 0.5pi
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
    scaling = tf.random.uniform([2], 0.8, 1, seed=seed)
    x_scaling = scaling[0]
    y_scaling = scaling[1]
    out_h = tf.cast(in_h * y_scaling, dtype=tf.int32)
    out_w = tf.cast(in_w * x_scaling, dtype=tf.int32)
    image = tf.image.random_crop(image, size=[out_h, out_w, 3], seed=seed)
    image = tf.image.resize(image, size=(in_h, in_w))

    # Random shearing
    x_shear = tf.random.uniform([1], minval=-0.1, maxval=0.1, dtype=tf.float32, seed=seed)[0]
    y_shear = tf.random.uniform([1], minval=-0.1, maxval=0.1, dtype=tf.float32, seed=seed)[0]
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
