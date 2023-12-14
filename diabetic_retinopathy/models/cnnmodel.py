from models.cnnblocks import cnn_block
import tensorflow as tf
import gin
@gin.configurable
def create_cnn_nets(input_shape, num_blocks, filters, kernel_size, dense_units, dropout_rate):
    """
    Builds a binary classification CNN network with multiple CNN blocks followed by a dense layer with dropout.

    Parameters:
        input_shape (tuple): Shape of the input images.
        num_blocks (int): Number of CNN blocks to be used.
        filters (int): Number of filters for the first CNN block (doubles with each block).
        kernel_size (tuple): Size of the kernel for the CNN blocks.
        dense_units (int): Number of units in the dense layer.
        dropout_rate (float): Dropout rate to be used before the dense layer.

    Returns:
        keras.Model: Constructed Keras model for binary classification.
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    for i in range(num_blocks):
        x = cnn_block(x, filters * (2 ** i), kernel_size)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(dense_units, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)  # Binary classification output

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model