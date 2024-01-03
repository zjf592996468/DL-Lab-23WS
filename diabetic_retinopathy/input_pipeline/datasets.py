import gin
import logging
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from diabetic_retinopathy.input_pipeline.preprocessing import preprocess, augment, resample


# create tfrecord and load datasets
# define TFRecord assist func
def _bytes_feature(value):
    """返回一个bytes_list从一个字符串 / 字节"""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """返回一个int64_list从一个布尔值/整数"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# define func to create tfrecorder files
def create_tfrecord(tfrd_path, img_dir, labels):
    with tf.io.TFRecordWriter(tfrd_path) as writer:
        # according to labels to read files
        for index, row in labels.iterrows():
            try:
                img_path = os.path.join(img_dir, row['Image name'] + '.jpg')
                img = open(img_path, 'rb').read()
                label = row['Retinopathy grade']
                # according to task to group labels into 2 groups
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


@gin.configurable
def load(name, data_dir, split_frac):
    if name == "idrid":
        logging.info(f"Preparing dataset {name}...")

        # 设置图片和标签的目录
        train_img_dir = os.path.join(data_dir, "images", "train")
        test_img_dir = os.path.join(data_dir, "images", "test")
        labels_dir = os.path.join(data_dir, "labels")

        # 读取标签文件

        # train_labels = pd.read_csv(os.path.join(labels_dir, "train.csv"), usecols=["Image name", "Retinopathy grade"])
        # # 进行分层抽样
        # train_labels, val_labels = train_test_split(train_labels, test_size=split_frac,
        #                                             stratify=train_labels['Retinopathy grade'])

        train_labels= pd.read_csv(os.path.join(labels_dir, "train.csv"), usecols=["Image name", "Retinopathy grade"])
        test_labels = pd.read_csv(os.path.join(labels_dir, "test.csv"), usecols=["Image name", "Retinopathy grade"])

        # 分割训练集和验证集
        val_size = int(len(train_labels) * split_frac)
        val_labels = train_labels[:val_size]
        train_labels = train_labels[val_size:]

        # 获取当前工作目录
        current_dir = os.getcwd()

        # 设置TFRecord文件的路径
        val_tfrd_path = os.path.join(current_dir, "val.tfrecord")
        train_tfrd_path = os.path.join(current_dir, "train.tfrecord")
        test_tfrd_path = os.path.join(current_dir, "test.tfrecord")

        # 创建TFRecord文件
        create_tfrecord(train_tfrd_path, train_img_dir, train_labels)
        create_tfrecord(val_tfrd_path, train_img_dir, val_labels)
        create_tfrecord(test_tfrd_path, test_img_dir, test_labels)

        # 读取TFRecord文件并创建原始数据集
        ds_train = tf.data.TFRecordDataset(train_tfrd_path).map(_parse_tfrd_function)
        ds_test = tf.data.TFRecordDataset(test_tfrd_path).map(_parse_tfrd_function)
        ds_val = tf.data.TFRecordDataset(val_tfrd_path).map(_parse_tfrd_function)

        label0_count = 0
        label1_count = 0

        for _, label in ds_train:
            label = label.numpy()
            if label == 0:
                label0_count += 1
            elif label == 1:
                label1_count += 1

        # this is for test ob train datasets is correct
        # print(f"Label 0 count: {label0_count}")
        # print(f"Label 1 count: {label1_count}")

        for image, label in ds_train.take(1):
            # get image shape
            shape = image.numpy().shape

        # 更新 ds_info 字典
        ds_info = {
            'train_size': label0_count + label1_count,
            'val_size': 413-(label0_count + label1_count),
            'test_size': 103,
            'label0_count': label0_count,
            'label1_count': label1_count,
            'shape': shape,
            'num_classes': 2
            # 其他信息
        }

        return prepare(ds_train, ds_val, ds_test, ds_info, seed=2023, batch_size=32, caching=True)

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
def prepare(ds_train, ds_val, ds_test, ds_info, seed, batch_size, caching):
    # Prepare training dataset
    ds_train = ds_train.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    ds_train = resample(ds_train)

    for image, label in ds_train.take(1):
        # get image shape
        shape = image.numpy().shape
    ds_info.update({

        'shape': shape,
    })

    if caching:
        ds_train = ds_train.cache()

    ds_train = ds_train.map(
        augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.shuffle(buffer_size=40, seed=seed)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat(-1)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare validation dataset
    ds_val = ds_val.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.batch(batch_size)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare test dataset
    ds_test = ds_test.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_val, ds_test, ds_info