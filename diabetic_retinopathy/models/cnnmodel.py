from models.cnnblocks import cnn_block
import tensorflow as tf
import gin
@gin.configurable
def create_cnn_nets(input_shape, num_classes, num_blocks, filters, kernel_size, dense_units, dropout_rate):
    """
    Builds a CNN network with multiple CNN blocks followed by a dense layer with dropout for classification.

    Parameters:
        input_shape (tuple): Shape of the input images.
        num_classes (int): Number of classes for the output layer.
        num_blocks (int): Number of CNN blocks to be used.
        filters (int): Number of filters for the first CNN block (doubles with each block).
        kernel_size (tuple): Size of the kernel for the CNN blocks.
        dense_units (int): Number of units in the dense layer.
        dropout_rate (float): Dropout rate to be used before the dense layer.

    Returns:
        keras.Model: Constructed Keras model.
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    for i in range(num_blocks):
        x = cnn_block(x, filters * (2 ** i), kernel_size)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(dense_units, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)  # 添加 Dropout 层
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
