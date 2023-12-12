import tensorflow as tf


def cnn_block(inputs, filters, kernel_size, use_batch_norm=True, max_pool=True):
    """
    Creates a CNN block with convolution, activation, optional batch normalization and max pooling.

    Parameters:
        inputs (Tensor): Input tensor for the CNN block.
        filters (int): Number of filters for the convolutional layer.
        kernel_size (tuple): Size of the kernel for the convolutional layer.
        use_batch_norm (bool): Whether to include batch normalization.
        max_pool (bool): Whether to include a max pooling layer.

    Returns:
        Tensor: Output tensor of the CNN block.
    """

    x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(inputs)
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    if max_pool:
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    return x
