import tensorflow as tf
import tensorflow_hub as hub


def transfermodel(input_shape, n_classes, trainable=False):
    """
    创建一个预训练的 EfficientNet V2 模型。

    参数:
    input_shape (tuple): 输入图像的尺寸，例如 (224, 224, 3)。
    n_classes (int): 输出类别的数量，默认为 1000（ImageNet 类别数）。
    trainable (bool): 是否对基础模型层进行训练。

    返回:
    model: 构建的 Keras 模型。
    """

    # EfficientNet V2 预训练模型的 TensorFlow Hub URL
    efficientnet_v2_url = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/classification/2"

    # Create Models
    model = tf.keras.Sequential([
        hub.KerasLayer(efficientnet_v2_url, input_shape=input_shape, trainable=trainable),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(n_classes)
    ])

    # Compile Model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
