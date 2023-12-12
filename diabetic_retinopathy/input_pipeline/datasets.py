import gin
import logging
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import pandas as pd

from input_pipeline.preprocessing import preprocess, augment, resample


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
def create_tfrecord(tfrd_path, img_dir, labels, group):
    with tf.io.TFRecordWriter(tfrd_path) as writer:
        # according to labels to read files
        for index, row in labels.iterrows():
            try:
                img_path = os.path.join(img_dir, row['Image name'] + '.jpg')
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
def load(name, data_dir, split_frac, tfrd_dir, group):
    if name == "idrid":
        logging.info(f"Preparing dataset {name}...")

        # data directories
        train_img_dir = os.path.join(data_dir, "images", "train")
        test_img_dir = os.path.join(data_dir, "images", "test")
        labels_dir = os.path.join(data_dir, "labels")

        # read label files, only read rows of "Image name" and "Retinopathy grade"
        train_labels = pd.read_csv(os.path.join(labels_dir, "train.csv"), usecols=["Image name", "Retinopathy grade"])
        test_labels = pd.read_csv(os.path.join(labels_dir, "test.csv"), usecols=["Image name", "Retinopathy grade"])

        # path to create TFRecord
        train_tfrd_path = os.path.join(tfrd_dir, "train.tfrecord")
        test_tfrd_path = os.path.join(tfrd_dir, "test.tfrecord")

        # create TFRecord files for origin train and test
        create_tfrecord(train_tfrd_path, train_img_dir, train_labels, group)
        create_tfrecord(test_tfrd_path, test_img_dir, test_labels, group)

        # read TFRecord files and create origin dataset
        train_data = tf.data.TFRecordDataset(train_tfrd_path).map(_parse_tfrd_function)
        ds_test = tf.data.TFRecordDataset(test_tfrd_path).map(_parse_tfrd_function)

        # # resample train_data
        # # 对训练数据做重采样，并在plot_path保存重采样前数据分布图表，输出重采样后的训练数据量和class数量
        # plot_path = os.path.join(tfrd_dir, 'class_distribution_before_resampling.png')
        # (train_data, train_samples, num_classes) = resample(train_data, plot_path)
        print("resample finished")
        logging.info(f"train data resampled")

        # split train and validation dataset with split_frac = 0.9
        train_samples = train_labels.shape[0]
        train_size = int(split_frac * train_samples)
        ds_train = train_data.take(train_size)
        ds_val = train_data.skip(train_size)
        print("dataset split finished")
        logging.info(f"split dataset into train and validation")

        # 构建数据集信息
        ds_info = {
            'train_size': train_size,
            'val_size': train_samples - train_size,
            'test_size': test_labels.shape[0],
            # 其他信息
            # 'num_classes': num_classes,
        }
        print("setup ds_info finished")
        logging.info(f"ds_info established")

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
    # Prepare training dataset
    ds_train = ds_train.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if caching:
        ds_train = ds_train.cache()
    ds_train = ds_train.map(
        augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # 将处理过后的图像信息录入ds_info
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

    ds_train = ds_train.shuffle(ds_info['train_size'] // 10)
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
