import gin
import tensorflow as tf
import random
import tensorflow_addons as tfa
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import logging


def check_imb(dataset):
    """check and plot imbalance situation, return num of classes and num of samples in each class"""
    # 获取原始数据，转换为 NumPy 数组
    lb = np.array([label.numpy() for _, label in dataset])

    # 获取原始数据的类别分布
    label_class, counts = np.unique(lb, return_counts=True)

    # 为每个类别设置不同的颜色
    colors = plt.cm.get_cmap('tab10')(np.arange(label_class.shape[0]))

    # 绘制柱状图
    bars = plt.bar(label_class, counts, color=colors, label='Class percentages')  # color='skyblue'
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution Before Resampling')

    # 设置 x 轴的标签为类别标签
    plt.xticks(label_class)

    # 在柱状图上显示数据标签
    for i, count in enumerate(counts):
        plt.text(label_class[i], count, str(count), ha='center', va='bottom')

    # 显示图例在左上角，并且里面包含每个类别的占比信息
    total_samples = lb.shape[0]
    percents = [count / total_samples * 100 for count in counts]
    legend_labels = [f'Class {label_class[i]}: {percents[i]:.2f}%' for i in range(label_class.shape[0])]
    plt.legend(bars, legend_labels)

    # 保存图表到文件
    plot_path = Path.cwd().parent.parent
    plt.savefig(plot_path / ('Class distribution before resampling' + '.png'))
    logging.info(f"Class distribution plot before resampling is drawn in {plot_path.resolve()}.")
    plt.close()  # 关闭图表，释放资源

    return label_class.shape[0], counts


def resample(dataset, ds_info):
    """resample the dataset to equal distribution and return it"""
    # method: rejection resample
    dataset_re = dataset.rejection_resample(
        class_func=lambda image, label: label,
        target_dist=[1.0/ds_info['num_classes']]*ds_info['num_classes'],
        seed=18)
    # 使用 map 删除多余的标签副本
    dataset_re = dataset_re.map(lambda extra_label, image_and_label: image_and_label)

    # # method: sample from datasets
    # # 根据标签拆分数据集
    # datasets_by_label = []
    # for label in range(ds_info['num_classes']):
    #     filtered_dataset = dataset.filter(lambda image, lbl: tf.equal(lbl, label))
    #     datasets_by_label.append(filtered_dataset)
    # # 使用 sample_from_datasets 进行重采样
    # weight = [1.0 / ds_info['num_classes']] * ds_info['num_classes']
    # dataset_re = tf.data.Dataset.sample_from_datasets(datasets_by_label, seed=18)

    # 更新重采样后训练集大小
    train_size_re = np.array([label.numpy() for _, label in dataset_re]).shape[0]
    ds_info.update({
        'train_size': train_size_re,
    })
    logging.info("train_size updated.")

    return dataset_re, ds_info


@gin.configurable
def preprocess(image, label, img_height, img_width):
    """Dataset preprocessing: Normalizing and resizing"""

    # Normalize image: `uint8` -> `float32`.
    image = tf.cast(image, tf.float32) / 255.

    # Resize image with pad to avoid distortions
    image = tf.image.resize_with_pad(image, img_height, img_width)

    return image, label


def augment(image, label):
    operations = ['Rotation90', 'Rotation180', 'Rotation270', 'Flippinglr', 'Flippingud', 'Cropping','Shearing']
    # Randomly choose one or more operations
    chosen_operations = random.sample(operations, k=random.randint(1, len(operations)))

    for operation in chosen_operations:
        if operation == 'Rotation90':
            image = tf.image.rot90(image)
        elif operation == 'Rotation180':
            image = tf.image.rot90(image, 2)
        elif operation == 'Rotation270':
            image = tf.image.rot90(image, 3)
        elif operation == 'Flippinglr':
            image = tf.image.flip_left_right(image)
        elif operation == 'Flippingud':
            image = tf.image.flip_up_down(image)
        elif operation == 'Cropping':
            # Randomly crop and resize back to 256x256
            cropped_size = [tf.random.uniform([], minval=180, maxval=256, dtype=tf.int32) for _ in range(2)]
            image = tf.image.random_crop(image, size=[cropped_size[0], cropped_size[1], 3])
            image = tf.image.resize(image, [256, 256])
        elif operation == 'Shearing':
            # Shearing using affine transformation, keeping image size constant
            shear_x = random.uniform(-0.3, 0.3)  # Shear magnitude along x-axis
            shear_y = random.uniform(-0.3, 0.3)  # Shear magnitude along y-axis
            image = tfa.image.transform(image, [1.0, shear_x, 0.0, shear_y, 1.0, 0.0, 0.0, 0.0],
                                        interpolation='NEAREST')
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

    # elif 'Cropping' in operation:  # crop the central region of the image
    #     image = tf.image.central_crop(image, central_frac)

    # elif 'Shearing' in operation:  # shear the image
    #     image = tf.image.shear(image)  # no such method in tf.image

    return image, label
