import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.python.keras.models import Sequential
from keras.applications import VGG16


def transfermodel(input_shape, n_classes, dropout_rate=0.5, dense_units=1024):
    # 加载预训练的 VGG16 模型，不包括顶层
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # 冻结 VGG16 模型的参数
    base_model.trainable = False

    model = Sequential([
        base_model,  # VGG16 模型作为基础
        GlobalAveragePooling2D(),  # 添加全局平均池化层
        Dense(dense_units, activation='relu'),  # 全连接层
        Dropout(dropout_rate),  # Dropout 层
        Dense(n_classes)  # 输出层，不使用激活函数，预期直接输出 logits
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model
