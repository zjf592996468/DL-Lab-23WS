from models.cnnblocks import cnn_block
import tensorflow as tf
import gin

@gin.configurable
def create_cnn_nets(input_shape, num_blocks, filters, kernel_size, dense_units, dropout_rate, l2_lambda):
    """
    Builds an advanced CNN model for binary classification with multiple CNN blocks and L2 regularization.
    Parameters:
        input_shape (tuple): Shape of the input images.
        num_blocks (int): Number of CNN blocks to be used.
        filters (int): Number of filters for the first CNN block.
        kernel_size (tuple): Size of the kernel for the CNN blocks.
        dense_units (int): Number of units in the dense layer.
        dropout_rate (float): Dropout rate to be used before the dense layer.
        l2_lambda (float): Lambda value for L2 regularization.
    Returns:
        keras.Model: Constructed Keras model for binary classification.
    """

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs


    # Xavier/Glorot initialization for each CNN block
    for i in range(num_blocks):
        x = cnn_block(x, filters * (2 ** i), kernel_size, l2_lambda)

    out = tf.keras.layers.GlobalAveragePooling2D()(x)
    out = tf.keras.layers.Dense(dense_units, activation=tf.nn.relu,
                                kernel_regularizer=tf.keras.regularizers.l2(l2_lambda),
                                kernel_initializer='glorot_uniform')(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(2, kernel_regularizer=tf.keras.regularizers.l2(l2_lambda),
                                    kernel_initializer='glorot_uniform')(out)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='cnn_like')