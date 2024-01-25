import numpy as np
import tensorflow as tf
from PIL import Image
import cv2


def grad_cam(model, image, category_index, layer_name):
    # load the model
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[:, category_index]

    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    gate_f = tf.cast(output > 0, 'float32')
    gate_r = tf.cast(grads > 0, 'float32')
    guided_grads = gate_f * gate_r * grads

    weights = tf.reduce_mean(guided_grads, axis=(0, 1))

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
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_RAINBOW)

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


'''Model: "vgg_like"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 256, 256, 3)]     0         

 conv2d (Conv2D)             (None, 256, 256, 43)      1204      

 conv2d_1 (Conv2D)           (None, 256, 256, 43)      16684     

 max_pooling2d (MaxPooling2  (None, 128, 128, 43)      0         
 D)                                                              

 conv2d_2 (Conv2D)           (None, 128, 128, 172)     66736     

 conv2d_3 (Conv2D)           (None, 128, 128, 172)     266428    

 max_pooling2d_1 (MaxPoolin  (None, 64, 64, 172)       0         
 g2D)                                                            

 global_average_pooling2d (  (None, 172)               0         
 GlobalAveragePooling2D)                                         

 dense (Dense)               (None, 32)                5536      

 dropout (Dropout)           (None, 32)                0         

 dense_1 (Dense)             (None, 2)                 66        

=================================================================
Total params: 356654 (1.36 MB)
Trainable params: 356654 (1.36 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
'''

'''Model: "cnn_like"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 256, 256, 3)]     0         

 conv2d (Conv2D)             (None, 256, 256, 8)       224       

 batch_normalization (Batch  (None, 256, 256, 8)       32        
 Normalization)                                                  

 max_pooling2d (MaxPooling2  (None, 128, 128, 8)       0         
 D)                                                              

 conv2d_1 (Conv2D)           (None, 128, 128, 16)      1168      

 batch_normalization_1 (Bat  (None, 128, 128, 16)      64        
 chNormalization)                                                

 max_pooling2d_1 (MaxPoolin  (None, 64, 64, 16)        0         
 g2D)                                                            

 conv2d_2 (Conv2D)           (None, 64, 64, 32)        4640      

 batch_normalization_2 (Bat  (None, 64, 64, 32)        128       
 chNormalization)                                                

 max_pooling2d_2 (MaxPoolin  (None, 32, 32, 32)        0         
 g2D)                                                            

 global_average_pooling2d (  (None, 32)                0         
 GlobalAveragePooling2D)                                         

 dense (Dense)               (None, 128)               4224      

 dropout (Dropout)           (None, 128)               0         

 dense_1 (Dense)             (None, 2)                 258       

=================================================================
Total params: 10738 (41.95 KB)
Trainable params: 10626 (41.51 KB)
Non-trainable params: 112 (448.00 Byte)
_________________________________________________________________
'''
