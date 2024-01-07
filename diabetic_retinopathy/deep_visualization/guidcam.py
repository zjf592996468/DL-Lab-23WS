import tensorflow as tf
import numpy as np


# 修改 ReLU 行为
# Define the guided ReLU activation function
@tf.custom_gradient
def guided_relu(x):
    def grad(dy):
        return tf.cast(dy > 0, "float32") * tf.cast(x > 0, "float32") * dy

    return tf.nn.relu(x), grad


# Function to recursively replace ReLU activations
def replace_relu_with_guided_relu(model):
    for layer in model.layers:
        if hasattr(layer, 'activation') and layer.activation == tf.keras.activations.relu:
            layer.activation = guided_relu
        # If the layer is a model itself, recursively replace its ReLU
        if hasattr(layer, 'layers'):
            replace_relu_with_guided_relu(layer)
    # After modifying the model, return it
    return model


def guided_grad_cam(model, image, category_index, layer_name):
    # 创建新的模型，以获取卷积层输出和最终预测
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

    # 用 Guided ReLU 替换标准的 ReLU
    model = replace_relu_with_guided_relu(grad_model)

    with tf.GradientTape() as tape:
        # 正向传播
        conv_outputs, predictions = model(image)

        # 确保 predictions 是张量
        if isinstance(predictions, list):
            predictions = predictions[0]

        # 计算 loss
        loss = predictions[:, category_index]

    # 计算梯度
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    # 计算 Guided Gradients
    gate_f = tf.cast(output > 0, 'float32')
    gate_r = tf.cast(grads > 0, 'float32')
    guided_grads = gate_f * gate_r * grads

    # 计算权重和类激活映射
    weights = tf.reduce_mean(guided_grads, axis=(0, 1))
    cam = np.ones(output.shape[0:2], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    # 归一化和生成热力图
    cam = np.maximum(cam, 0)
    heatmap = (cam - cam.min()) / (cam.max() - cam.min())

    return heatmap
