import tensorflow as tf
import tensorflow_hub as hub
from pathlib import Path



def transfermodel(input_shape, n_classes, trainable=False):
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

    # 创建模型
    model = tf.keras.Sequential([
        hub.KerasLayer(str(model_path), input_shape=input_shape, trainable=trainable),
        tf.keras.layers.Dense(units=1024,activation=tf.nn.relu,kernel_initializer=tf.keras.initializers.glorot_uniform),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(n_classes)
    ])

    # 编译模型
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model