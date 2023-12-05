from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def create_and_compile_cnn_model():
    model = Sequential([
        # 第一层卷积，使用32个3x3的卷积核，激活函数为ReLU
        Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        MaxPooling2D(2, 2),

        # 第二层卷积，使用64个3x3的卷积核，激活函数为ReLU
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        # 第三层卷积，使用128个3x3的卷积核，激活函数为ReLU
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        # 将卷积层输出扁平化为一维向量
        Flatten(),

        # 全连接层，128个节点，激活函数为ReLU
        Dense(128, activation='relu'),

        # Dropout层，以防止过拟合，丢弃率为0.3
        Dropout(0.3),

        # 输出层，2个节点（每个类别一个），激活函数为sigmoid
        Dense(2, activation='sigmoid')
    ])

    # 编译模型
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model