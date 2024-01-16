import tensorflow as tf
import numpy as np


# Define the guided ReLU activation function
@tf.custom_gradient
def guided_relu(x):
    def grad(dy):
        return tf.cast(dy > 0, "float32") * tf.cast(x > 0, "float32") * dy

    return tf.nn.relu(x), grad


# Function to recursively replace ReLU activations
def replace_relu_with_guided_relu(model):
    for layer in model.layers:
        if hasattr(layer, 'activation') and layer.activation == tf.nn.relu:
            layer.activation = guided_relu
        # If the layer is a model, recursively replace its ReLU
        if hasattr(layer, 'layers'):
            replace_relu_with_guided_relu(layer)

    # Return after modifying the model
    return model


def guided_grad_cam(model, image, category_index, layer_name):
    # Create a new model to get the convolutional layer output and final prediction
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

    # Replacing Standard ReLUs with Guided ReLUs
    model = replace_relu_with_guided_relu(grad_model)

    with tf.GradientTape() as tape:
        # Forward propagation
        conv_outputs, predictions = model(image)

        # Make sure predictions are tensors
        if isinstance(predictions, list):
            predictions = predictions[0]

        # Calculate loss
        loss = predictions[:, category_index]

    # Calculate the gradient
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    # Calculate Guided Gradients
    gate_f = tf.cast(output > 0, 'float32')
    gate_r = tf.cast(grads > 0, 'float32')
    guided_grads = gate_f * gate_r * grads

    # Calculate weights and class activation mappings
    weights = tf.reduce_mean(guided_grads, axis=(0, 1))
    cam = np.ones(output.shape[0:2], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    # Normalisation and generation of heat maps
    cam = np.maximum(cam, 0)
    heatmap = (cam - cam.min()) / (cam.max() - cam.min())

    return heatmap
