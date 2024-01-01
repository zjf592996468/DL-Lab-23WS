import tensorflow as tf
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Sequential
from keras.applications.vgg16 import VGG16

def transfermodel(input_shape=(224, 224, 3), n_classes=2, dropout_rate=0.5, dense_units=1024):
    # 加载预训练的 VGG16 模型，不包括顶层
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    model = Sequential([
        base_model,  # VGG16 模型作为基础
        GlobalAveragePooling2D(),  # 添加全局平均池化层
        Dense(dense_units, activation='relu'),  # 全连接层
        Dropout(dropout_rate),  # Dropout 层
        Dense(n_classes, activation='softmax' if n_classes > 1 else 'sigmoid')  # 输出层
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False) if n_classes > 1 else 'binary_crossentropy',
                  metrics=['accuracy'])
    return model