import tensorflow as tf
import numpy as np

class ConfusionMatrix(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='confusion_matrix', **kwargs):
        super(ConfusionMatrix, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.confusion_matrix = self.add_weight(name='conf_matrix', shape=(num_classes, num_classes),
                                                initializer='zeros', dtype=tf.int32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, self.dtype)
        y_pred = tf.cast(tf.argmax(y_pred, axis=1), self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
        else:
            sample_weight = tf.ones_like(y_true)

        current_cm = tf.math.confusion_matrix(y_true, y_pred, self.num_classes, weights=sample_weight, dtype=tf.int32)
        return self.confusion_matrix.assign_add(current_cm)

    def result(self):
        return tf.identity(self.confusion_matrix)

    def reset_states(self):
        tf.keras.backend.set_value(self.confusion_matrix, np.zeros((self.num_classes, self.num_classes), dtype=np.int32))


