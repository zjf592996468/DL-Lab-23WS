import gin
import logging
import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path
import pandas as pd

from input_pipeline.preprocessing import preprocess, augment, check_imb, resample


def _bytes_feature(value):
    """TFRecord assist func: Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """TFRecord assist func: Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_tfrecord(tfrd_path, img_dir, labels, group):
    """Create a tfrecord at 'tfrd_path' with datas from 'img_dir' and 'labels'"""
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


def _parse_tfrd_function(example_proto):
    """Parse the example_proto from TensorFlow Examples"""
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
        logging.info(f"Dataset is divided into train and validation with rate = {split_frac}.")

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

        # check and plot ds_train imbalance situation, get num of classes and num of samples in each class
        num_classes, counts = check_imb(ds_train, method=False)

        # 构建数据集信息
        ds_info = {
            'train_size': train_size,
            'val_size': train_labels.shape[0] - train_size,
            'test_size': test_labels.shape[0],
            'num_classes': num_classes,
        }

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

        ds_train = ds_train.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        ds_val = ds_val.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        ds_test = ds_test.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

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
        preprocess, num_parallel_calls=tf.data.AUTOTUNE)

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

    # resample ds_train
    ds_train, ds_info, re_method = resample(ds_train, ds_info)
    check_imb(ds_train, re_method)

    if caching:
        ds_train = ds_train.cache()
    ds_train = ds_train.map(
        augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.shuffle(ds_info['train_size'] // 10)  # todo: Q: will here with smaller num better?
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat(-1)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
    logging.info("ds_train prepared.")

    # Prepare validation dataset
    ds_val = ds_val.map(
        preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds_val = ds_val.batch(batch_size)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.AUTOTUNE)
    logging.info("ds_val prepared.")

    # Prepare test dataset
    ds_test = ds_test.map(
        preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
    logging.info("ds_test prepared.")

    return ds_train, ds_val, ds_test, ds_info
