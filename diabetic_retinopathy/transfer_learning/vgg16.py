import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.python.keras.models import Sequential
from keras.applications import VGG16


def transfermodel(input_shape, n_classes, dropout_rate=0.5, dense_units=1024):
    # Load the pre-trained VGG16 model, excluding the top layer
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freezing the parameters of the VGG16 model
    base_model.trainable = False

    model = Sequential([
        base_model,  # VGG16 as basic
        GlobalAveragePooling2D(),  # GAP layer
        Dense(dense_units, activation='relu'),  # Full connection layer
        Dropout(dropout_rate),  # Dropout layer
        Dense(n_classes)  # Output layer with no activation func, output prediction from logits
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model
