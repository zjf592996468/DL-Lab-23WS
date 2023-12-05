import gin
import tensorflow as tf
from PIL import Image, ImageOps

@gin.configurable
def preprocess(image, label, img_height=256, img_width=256):
    """Dataset preprocessing: Normalizing and resizing"""

    # Normalize image: `uint8` -> `float32`.
    tf.cast(image, tf.float32) / 255.

    # Resize image
    image = tf.image.resize(image, size=(img_height, img_width))

    return image, label

def augment(image, label):
    """Data augmentation"""
    # 水平翻转图像
    #mirrored_image = ImageOps.mirror(image)
    #return mirrored_image, label
    return image,label