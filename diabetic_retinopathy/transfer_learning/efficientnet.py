from pathlib import Path
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.models import Sequential


def transfermodel(input_shape, n_classes, dense_units=1024, dropout=0.5, trainable=False):
    """
    创建一个预训练的 EfficientNet V2 模型。

    参数:
    input_shape (tuple): 输入图像的尺寸，例如 (224, 224, 3)。
    n_classes (int): 输出类别的数量。
    trainable (bool): 是否对基础模型层进行训练。

    返回:
    model: 构建的 Keras 模型。
    """

    # 当前脚本所在目录
    current_dir = Path(__file__).parent

    # 模型所在的路径
    model_path = current_dir / "archive"

    model = Sequential([
        hub.KerasLayer(str(model_path), input_shape=input_shape, trainable=trainable),
        Dense(dense_units, activation='relu'),  # 调整神经元数量
        Dropout(dropout),
        Dense(units=n_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model
