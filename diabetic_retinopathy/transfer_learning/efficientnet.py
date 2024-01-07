import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.python.keras.layers import Dense, Dropout,GlobalAveragePooling2D
from tensorflow.python.keras.models import Sequential


def transfermodel(input_shape, n_classes, dropout_rate=0.5, dense_units=1024):  # 减少神经元数量
    efficientnet_url = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"

    model = Sequential([
        hub.KerasLayer(efficientnet_url, input_shape=input_shape, trainable=False),

        Dense(dense_units, activation='relu'),  # 调整神经元数量
        Dropout(dropout_rate),
        Dense(n_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model
