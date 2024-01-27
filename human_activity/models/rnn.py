import gin
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, GRU
from keras.regularizers import l2
from absl.flags import FLAGS
import tensorflow as tf

@gin.configurable
def create_rnn(ds_info, lstm_units, dense_units, dropout_rate, regularization_factor):
    """
    Build a Bidirectional LSTM model for action recognition.
    Parameters:
    ds_info (dict): Dataset information, including 'features_shape' and 'num_classes'.
    lstm_units (int): Number of units in the LSTM layers.
    dense_units (int): Number of units in the dense layer.
    dropout_rate (float): Dropout rate.
    regularization_factor (float): L2 regularization factor.
    Returns:
    model: Constructed Keras model.
    """
    model = Sequential()

    # LSTM layer/Bidirectional LSTM layer/GRU/Bidirectional GRU
    if FLAGS.layer == 'Bidirectional LSTM':
        model.add(Bidirectional(LSTM(lstm_units,
                                     return_sequences=True,
                                     kernel_initializer=tf.keras.initializers.glorot_uniform,
                                     kernel_regularizer=l2(regularization_factor),
                                     input_shape=ds_info['features_shape'])))

    elif FLAGS.layer == 'LSTM':
        model.add(LSTM(lstm_units,
                       return_sequences=True,
                       kernel_regularizer=l2(regularization_factor),
                       kernel_initializer=tf.keras.initializers.glorot_uniform,
                       input_shape=ds_info['features_shape']))

    elif FLAGS.layer == 'Bidirectional GRU':
        model.add(Bidirectional(GRU(lstm_units,
                                    return_sequences=True,
                                    kernel_regularizer=l2(regularization_factor),
                                    kernel_initializer=tf.keras.initializers.glorot_uniform,
                                    input_shape=ds_info['features_shape'])))

    elif FLAGS.layer == 'GRU':
        model.add(GRU(lstm_units,
                      return_sequences=True,
                      kernel_regularizer=l2(regularization_factor),
                      kernel_initializer=tf.keras.initializers.glorot_uniform,
                      input_shape=ds_info['features_shape']))

    else:
        raise ValueError("Invalid layer type. Choose from 'Bidirectional LSTM', 'LSTM', 'Bidirectional GRU', 'GRU'.")

    # Dropout layer
    model.add(Dropout(dropout_rate))

    # Dense layer
    model.add(Dense(dense_units, activation='relu',
                    kernel_initializer=tf.keras.initializers.glorot_uniform,
                    kernel_regularizer=l2(regularization_factor)))

    # Output layer
    model.add(Dense(ds_info['num_acts'], activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Build the model
    model.build(input_shape=(None, *ds_info['features_shape']))

    return model