import gin
import logging
import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from input_pipeline.preprocessing import preprocess, augment


# create tfrecord and load datasets
# define TFRecord assist func
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# define func to create tfrecorder files
def create_tfrecord(tfrd_path, img_dir, labels, group):
    """Create a tfrecord at tfrd_path with datas img_dir and labels"""
    with tf.io.TFRecordWriter(str(tfrd_path)) as writer:
        # according to labels to read files
        for index, row in labels.iterrows():
            try:
                img_path = img_dir / (row['Image name'] + '.jpg')
                img = open(img_path, 'rb').read()
                label = row['Retinopathy grade']

                # according to task to group labels into 2 groups
                if group:
                    label = 0 if label in [0, 1] else 1

                feature = {
                    'image': _bytes_feature(img),
                    'label': _int64_feature(label),
                }

                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
            except FileNotFoundError:
                print(f"File not found: {img_path}")
            except Exception as e:
                print(f"Error processing file {img_path}: {e}")


# parse tfrecord files
def _parse_tfrd_function(example_proto):
    """Parse the example_proto"""
    # 定义你的 `features`
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    # 从 proto 解析出 features
    example = tf.io.parse_single_example(example_proto, feature_description)

    # 对 JPEG 图像数据进行解码
    example['image'] = tf.io.decode_jpeg(example['image'], channels=3)

    return example['image'], example['label']


def check_imb(dataset):
    """check and plot imbalance situation, return number of classes and number of samples in each class"""
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


@gin.configurable
def load(name, data_dir, split_frac, group):
    """Load the dataset"""
    if name == "idrid":
        logging.info(f"Preparing dataset {name}...")

        # Data directory path
        train_img_dir = Path(data_dir) / "images" / "train"
        test_img_dir = Path(data_dir) / "images" / "test"
        labels_dir = Path(data_dir) / "labels"
        tfrd_dir = Path.cwd().parent.parent

        # read label files, only read rows of "Image name" and "Retinopathy grade"
        train_labels = pd.read_csv(Path(labels_dir) / "train.csv", usecols=["Image name", "Retinopathy grade"])
        test_labels = pd.read_csv(Path(labels_dir) / "test.csv", usecols=["Image name", "Retinopathy grade"])

        # path to create TFRecord
        train_tfrd_path = tfrd_dir / "train.tfrecord"
        val_tfrd_path = tfrd_dir / "val.tfrecord"
        test_tfrd_path = tfrd_dir / "test.tfrecord"

        # split train and validation dataset with split_frac = 0.9
        train_size = int(split_frac * train_labels.shape[0])
        train_dataset = train_labels[:train_size]
        val_dataset = train_labels[train_size:]
        logging.info("Dataset is divided into train and validation.")

        # create TFRecord files for origin train and test
        create_tfrecord(train_tfrd_path, train_img_dir, train_dataset, group)
        create_tfrecord(val_tfrd_path, train_img_dir, val_dataset, group)
        create_tfrecord(test_tfrd_path, test_img_dir, test_labels, group)
        logging.info(f"Tfrecord files are created in {tfrd_dir.resolve()}.")

        # read TFRecord files and create origin dataset
        ds_train = tf.data.TFRecordDataset(train_tfrd_path).map(_parse_tfrd_function)
        ds_val = tf.data.TFRecordDataset(val_tfrd_path).map(_parse_tfrd_function)
        ds_test = tf.data.TFRecordDataset(test_tfrd_path).map(_parse_tfrd_function)
        logging.info("Train, val and test datasets are created from tfrecord.")

        # check and plot ds_train imbalance situation, get num of classes and num od samples in each class
        (num_classes, counts) = check_imb(ds_train)

        # 构建数据集信息
        ds_info = {
            'train_size': train_size,
            'val_size': train_labels.shape[0] - train_size,
            'test_size': test_labels.shape[0],
            'num_classes': num_classes,
        }
        logging.info("ds_info recorded.")

        return prepare(ds_train, ds_val, ds_test, ds_info)

    elif name == "eyepacs":
        logging.info(f"Preparing dataset {name}...")
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            'diabetic_retinopathy_detection/btgraham-300',
            split=['train', 'validation', 'test'],
            shuffle_files=True,
            with_info=True,
            data_dir=data_dir
        )

        def _preprocess(img_label_dict):
            return img_label_dict['image'], img_label_dict['label']

        ds_train = ds_train.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_val = ds_val.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return prepare(ds_train, ds_val, ds_test, ds_info)

    elif name == "mnist":
        logging.info(f"Preparing dataset {name}...")
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            'mnist',
            split=['train[:90%]', 'train[90%:]', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
            data_dir=data_dir
        )

        return prepare(ds_train, ds_val, ds_test, ds_info)

    else:
        raise ValueError


@gin.configurable
def prepare(ds_train, ds_val, ds_test, ds_info, batch_size, caching):
    """Prepare the dataset for training, validation and test"""
    # Prepare training dataset
    ds_train = ds_train.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # update ds_info
    # 使用 take(1) 获取数据集中的一个元素
    sample_element = ds_train.take(1)
    # 直接获取第一个元素的图像形状
    image, label = next(iter(sample_element))
    img_height, img_width, img_channels = image.shape
    ds_info.update({
        'shape': (img_height, img_width, img_channels),
        'img_height': img_height,
        'img_width': img_width,
        'img_channels': img_channels,
    })
    logging.info("ds_info updated.")

    # # resample ds_train
    # ds_train_re = ds_train.rejection_resample(
    #     class_func=lambda image, label: label,
    #     target_dist=[1.0/ds_info['num_classes']]*ds_info['num_classes'],
    #     seed=18)
    # # 使用 map 删除多余的标签副本
    # ds_train = ds_train_re.map(lambda extra_label, image_and_label: image_and_label)
    # # 使用check_img查看分布
    # (num_classes_re, counts_re) = check_imb(ds_train)
    # # debug,显示拒绝重采样后的前几个样本
    # for features, labels in ds_train.take(10):
    #     print(labels.numpy())
    # logging.info("ds_train resampled.")
    # # 更新重采样后训练集大小
    # train_size_re = sum(1 for _ in ds_train)
    # print(train_size_re)
    # ds_info.update({
    #     'train_size': train_size_re,
    # })
    # logging.info("train_size updated.")

    if caching:
        ds_train = ds_train.cache()
    ds_train = ds_train.map(
        augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.shuffle(ds_info['train_size'] // 10)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat(-1)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
    logging.info("ds_train prepared.")

    # Prepare validation dataset
    ds_val = ds_val.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.batch(batch_size)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)
    logging.info("ds_val prepared.")

    # Prepare test dataset
    ds_test = ds_test.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
    logging.info("ds_test prepared.")

    return ds_train, ds_val, ds_test, ds_info
