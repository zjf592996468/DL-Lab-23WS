from pathlib import Path
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.models import Sequential
from absl.flags import FLAGS


def transfermodel(input_shape, n_classes, dense_units=1024, dropout=0.5):
    """
    Create a pretrained EfficientNet V2 model.

    Parameters:
    input_shape (tuple): The size of the input image, e.g., (224, 224, 3).
    n_classes (int): The number of output classes.
    trainable (bool): Whether to train the base model layers.

    Returns:
    model: The constructed Keras model.
    """

    # The directory of the current script
    current_dir = Path(__file__).parent

    # Path to the model
    model_path = current_dir / "archive"

    if FLAGS.multi_class:
        model = Sequential([
            hub.KerasLayer(str(model_path), input_shape=input_shape, trainable=False),
            Dense(dense_units, activation='relu'),
            Dropout(dropout),
            Dense(units=1)
        ])
    else:
        model = Sequential([
            hub.KerasLayer(str(model_path), input_shape=input_shape, trainable=False),
            Dense(dense_units, activation='relu'),
            Dropout(dropout),
            Dense(units=n_classes)
        ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model
