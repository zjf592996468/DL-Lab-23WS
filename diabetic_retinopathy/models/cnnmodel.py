from models.cnnblocks import cnn_block
import tensorflow as tf
import gin
import numpy as np
from absl.flags import FLAGS


@gin.configurable
def create_cnn_nets(ds_info, num_blocks, filters, kernel_size, dense_units, dropout_rate, seed, l2_lambda):
    """
    Builds an advanced CNN model for binary classification with multiple CNN blocks and L2 regularization.
    Parameters:
        ds_info: Dataset info dictionary
        num_blocks (int): Number of CNN blocks to be used.
        filters (int): Number of filters for the first CNN block.
        kernel_size (tuple): Size of the kernel for the CNN blocks.
        dense_units (int): Number of units in the dense layer.
        dropout_rate (float): Dropout rate to be used before the dense layer.
        # pick up what you like
        initializer = @tf.keras.initializers.HeNormal
        #initializer = @tf.keras.initializers.glorot_uniform(seed=2023)
        #initializer = @tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=2023)
        #initializer = @tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=2023)
        seed(int): Random seed to get reproducible results
        l2_lambda (float): Lambda value for L2 regularization.
    Returns:
        keras.Model: Constructed Keras model for binary classification.
    """

    inputs = tf.keras.Input(shape=ds_info['shape'])
    x = inputs
    label0_count = ds_info['class0_counts_re']
    label1_count = ds_info['class1_counts_re']

    # Calculate the initial bias
    initial_bias_value = np.log([label0_count / label1_count])

    # Initialization for each CNN block
    for i in range(num_blocks):
        x = cnn_block(x, filters * (2 ** i), kernel_size, l2_lambda)

    # GlobalMaxPolling layer
    out = tf.keras.layers.GlobalMaxPooling2D()(x)

    # Dense layer
    out = tf.keras.layers.Dense(dense_units,
                                activation=tf.nn.relu,
                                kernel_regularizer=tf.keras.regularizers.l2(l2_lambda),
                                kernel_initializer=tf.keras.initializers.glorot_uniform(seed))(out)

    # Dropout layer
    out = tf.keras.layers.Dropout(dropout_rate)(out)

    # Output layer
    if FLAGS.multi_class and not FLAGS.classification:
        # Use regression for multi-classification
        outputs = tf.keras.layers.Dense(units=1,
                                        kernel_regularizer=tf.keras.regularizers.l2(l2_lambda),
                                        kernel_initializer=tf.keras.initializers.glorot_uniform(seed))(out)
    else:

        # can also use this for multi-classification
        outputs = tf.keras.layers.Dense(units=ds_info['num_classes'],
                                        kernel_regularizer=tf.keras.regularizers.l2(l2_lambda),
                                        kernel_initializer=tf.keras.initializers.glorot_uniform(seed),
                                        bias_initializer=tf.keras.initializers.Constant(initial_bias_value))(out)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='cnn_like')
