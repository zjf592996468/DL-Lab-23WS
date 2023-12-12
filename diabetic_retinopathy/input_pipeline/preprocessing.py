import gin
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_imb(label, plot_path, plot_name):
    """plot the distribution of labels and return labels' class"""
    # 获取原始数据的类别分布
    unique, counts = np.unique(label, return_counts=True)

    # 绘制柱状图
    plt.bar(unique, counts)  # color='skyblue'
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title(plot_name)

    # 设置 x 轴的标签为类别标签
    plt.xticks(unique)

    # 在柱状图上显示数据标签
    for i, count in enumerate(counts):
        plt.text(unique[i], count, str(count), ha='center', va='bottom')

    # 保存图表到文件（如果提供了保存路径）
    if plot_path:
        plt.savefig(os.path.join(plot_path, plot_name + '.png'))
        plt.close()  # 关闭图表，释放资源
    else:
        plt.show()
        plt.close()  # 关闭图表，释放资源

    return unique


@gin.configurable
def preprocess(image, label, img_height, img_width):
    """Dataset preprocessing: Normalizing and resizing"""

    # Normalize image: `uint8` -> `float32`.
    image = tf.cast(image, tf.float32) / 255.

    # Resize image with pad to avoid distortions
    image = tf.image.resize_with_pad(image, img_height, img_width)

    return image, label


@gin.configurable()
def augment(image, label, operation):
    """Data augmentation"""
    if 'Rotation90' in operation:  # rotate 90 degree
        image = tf.image.rot90(image)

    elif 'Rotation180' in operation:  # rotate 180 degree
        image = tf.image.rot90(image, 2)

    elif 'Rotation270' in operation:  # rotate 270 degree
        image = tf.image.rot90(image, 3)

    elif 'Flippinglr' in operation:  # flip left and right
        image = tf.image.flip_left_right(image)

    elif 'Flippingud' in operation:  # flip up and down
        image = tf.image.flip_up_down(image)

    elif 'Cropping' in operation:  # crop the central region of the image
        image = tf.image.central_crop(image, central_frac=0.5)

    # elif 'Shearing' in operation:  # shear the image
    #     image = tf.image.shear(image)  # no such method in tf.image

    return image, label