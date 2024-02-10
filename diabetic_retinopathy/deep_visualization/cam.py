import numpy as np
import tensorflow as tf
from PIL import Image
import cv2


def grad_cam(model, image, category_index, layer_name):
    # load the model
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

    # calculate the grad with GradientTape
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[:, category_index]

    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    # we only have interest in positive influence
    gate_f = tf.cast(output > 0, 'float32')
    gate_r = tf.cast(grads > 0, 'float32')
    guided_grads = gate_f * gate_r * grads

    weights = tf.reduce_mean(guided_grads, axis=(0, 1))

    # plot heatmap
    cam = np.ones(output.shape[0:2], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = np.maximum(cam, 0)
    heatmap = (cam - cam.min()) / (cam.max() - cam.min())

    return heatmap


def overlay_heatmap(orig_image, heatmap, threshold, alpha):
    # Assuming orig_image is a PIL Image and has been normalized (0-1 range)
    orig_array = np.array(orig_image) * 255
    orig_array = orig_array.astype(np.uint8)

    # Resize heatmap to match original image size
    heatmap_resized = cv2.resize(heatmap, (orig_array.shape[1], orig_array.shape[0]))

    # Convert heatmap to color using a colormap
    heatmap_color = cv2.applyColorMap(np.uint8(-255 * heatmap_resized), cv2.COLORMAP_JET)

    # Create a mask where the heatmap is above the threshold
    mask = heatmap_resized > threshold

    # Create an overlay image, ensuring the data type is uint8
    overlay_image = np.zeros_like(orig_array, dtype=np.uint8)

    # Apply the mask to combine the original image and the heatmap
    overlay_image[~mask] = orig_array[~mask]

    # Adjust the alpha (transparency) of the heatmap overlay
    overlay_image[mask] = cv2.addWeighted(orig_array[mask], 1 - alpha, heatmap_color[mask], alpha, 0)

    # Convert the overlay image to uint8 type if not already
    overlay_image = np.clip(overlay_image, 0, 255).astype(np.uint8)

    # Return a PIL Image of the overlay image
    return Image.fromarray(overlay_image)


'''Model: "cnn_like"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 256, 256, 3)]     0         
                                                                 
 conv2d (Conv2D)             (None, 256, 256, 19)      532       
                                                                 
 batch_normalization (BatchN  (None, 256, 256, 19)     76        
 ormalization)                                                   
                                                                 
 max_pooling2d (MaxPooling2D  (None, 128, 128, 19)     0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 128, 128, 38)      6536      
                                                                 
 batch_normalization_1 (Batc  (None, 128, 128, 38)     152       
 hNormalization)                                                 
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 64, 64, 38)       0         
 2D)                                                             
                                                                 
 conv2d_2 (Conv2D)           (None, 64, 64, 76)        26068     
                                                                 
 batch_normalization_2 (Batc  (None, 64, 64, 76)       304       
 hNormalization)                                                 
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 32, 32, 76)       0         
 2D)                                                             
                                                                 
 global_max_pooling2d (Globa  (None, 76)               0         
 lMaxPooling2D)                                                  
                                                                 
 dense (Dense)               (None, 33)                2541      
                                                                 
 dropout (Dropout)           (None, 33)                0         
                                                                 
 dense_1 (Dense)             (None, 2)                 68        
                                                                 
=================================================================
'''
