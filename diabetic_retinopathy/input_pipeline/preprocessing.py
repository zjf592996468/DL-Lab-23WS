import gin
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# must run imbalanced-learn==0.7.0 scikit-learn==0.23.2 numpy==1.23.4
from imblearn.over_sampling import RandomOverSampler


def resample(dataset, plot_path):
    """Dataset resampling: check and output data imbalance, then resample dataset"""
    # 获取原始数据，转换为 NumPy 数组
    X = np.array([image.numpy() for image, _ in dataset])
    y = np.array([label.numpy() for _, label in dataset])

    # check data imbalance and plot imbalance situation
    # 获取原始数据的类别分布
    unique_original, counts_original = np.unique(y, return_counts=True)

    # 绘制柱状图
    plt.bar(unique_original, counts_original)  # color='skyblue'
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution Before Resampling')

    # 设置 x 轴的标签为类别标签
    plt.xticks(unique_original)

    # 在柱状图上显示数据标签
    for i, count in enumerate(counts_original):
        plt.text(unique_original[i], count, str(count), ha='center', va='bottom')

    # 保存图表到文件（如果提供了保存路径）
    if plot_path:
        plt.savefig(plot_path)
        plt.close()  # 关闭图表，释放资源
    else:
        plt.show()
        plt.close()  # 关闭图表，释放资源

    # random resample data
    # 使用 RandomOverSampler 进行重采样
    ros = RandomOverSampler(random_state=18)  # 定义重采样器
    X_flat = X.reshape(X.shape[0], -1)  # 将图片数组转换成二维数组
    X_resampled_flat, y_resampled = ros.fit_resample(X_flat, y)  # 重采样
    X_resampled = X_resampled_flat.reshape(-1, X.shape[1], X.shape[2], X.shape[3])  # 将二维图片数组还原

    # 将结果转换回 TensorFlow Dataset
    resampled_ds = tf.data.Dataset.from_tensor_slices((X_resampled, y_resampled))

    return resampled_ds, y_resampled.shape[0], unique_original.shape[0]


@gin.configurable
def preprocess(image, label, img_height, img_width):
    """Dataset preprocessing: Normalizing and resizing"""

    # Normalize image: `uint8` -> `float32`.
    image = tf.cast(image, tf.float32) / 255.

    # Resize image with pad to avoid distortions
    image = tf.image.resize_with_pad(image, img_height, img_width)

    return image, label


@gin.configurable()
def augment(image, label, operation, central_frac):
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
        image = tf.image.central_crop(image, central_frac)

    # elif 'Shearing' in operation:  # shear the image
    #     image = tf.image.shear(image)  # no such method in tf.image

    return image, label
