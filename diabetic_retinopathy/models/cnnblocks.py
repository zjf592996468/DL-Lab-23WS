import tensorflow as tf
import gin
import numpy as np


@gin.configurable
def cnn_block(inputs, filters, kernel_size, l2_lambda, seed, pool_size=(2, 2)):
    """
    Creates a CNN block with convolution, activation, batch normalization, max pooling, and L2 regularization.
    Parameters:

        inputs (Tensor): Input tensor for the CNN block.
        filters (int): Number of filters for the convolutional layer.
        kernel_size (tuple): Size of the kernel for the convolutional layer.
        pool_size (tuple): Size of the pooling window.
        l2_lambda (float): Lambda value for L2 regularization.
        seed (int): Random seed
        # pick up what you like
        initializer = @tf.keras.initializers.HeNormal
        #initializer = @tf.keras.initializers.glorot_uniform(seed=2023)
        #initializer = @tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=2023)
        #initializer = @tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=2023)


    Returns:
        Tensor: Output tensor of the CNN block.

    """

    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size,
        padding='same',
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(l2_lambda),
        kernel_initializer=tf.keras.initializers.glorot_uniform(seed)
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size)(x)
    return x
