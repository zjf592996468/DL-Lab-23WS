import gin
import logging
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import pandas as pd
from input_pipeline.preprocessing import preprocess, augment


def _bytes_feature(value):
    """返回一个bytes_list从一个字符串 / 字节"""
    # 如果是 EagerTensor，转换为numpy
    if isinstance(value, tf.Tensor):
        value = value.numpy()
    # 如果是字符串，首先编码
    if isinstance(value, str):
        value = value.encode()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """返回一个int64_list从一个布尔值/整数"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def preprocess_image(image_data):
    # 解码 JPEG 图像
    image = tf.image.decode_jpeg(image_data, channels=3)
    # 改变图像大小
    image = tf.image.resize_with_pad(image, 256, 256)
    # 归一化
    tf.cast(image, tf.float32) / 255.
    # 转换回字节
    image = tf.io.encode_jpeg(tf.cast(image, tf.uint8))
    return image.numpy()

def create_tfrecord(data_dir, df, output_file):
    with tf.io.TFRecordWriter(output_file) as writer:
        for index, row in df.iterrows():
            try:
                image_file = os.path.join(data_dir, row['Image name'] + '.jpg')
                grade = row['Retinopathy grade']
                label = 0 if grade in [0, 1] else 1
                with open(image_file, 'rb') as fid:
                    image_data = fid.read()
                #预处理图像
                image_data = preprocess_image(image_data)

                feature = {
                    'image': _bytes_feature(image_data),
                    'label': _int64_feature(label)
                }

                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
            except FileNotFoundError:
                print(f"File not found: {image_file}")
            except Exception as e:
                print(f"Error processing file {image_file}: {e}")

def _parse_function(proto):
    # 定义你的 `features`
    keys_to_features = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    # 从 proto 解析出 features
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)

    # 对 JPEG 图像数据进行解码
    parsed_features['image'] = tf.io.decode_jpeg(parsed_features['image'])


    return parsed_features['image'], parsed_features['label']


def create_dataset(filepath):
    # 这个数据集从TFRecord文件中读取数据
    dataset = tf.data.TFRecordDataset(filepath)
    # 映射解析函数
    dataset = dataset.map(_parse_function)
    return dataset


@gin.configurable
def load(name, data_dir):
    if name == "idrid":
        logging.info(f"Preparing dataset {name}...")
        # define the path
        train_csv_path = os.path.join(data_dir, "labels", "train.csv")
        test_csv_path = os.path.join(data_dir, "labels", "test.csv")
        test_image_path=os.path.join(data_dir, "images", "test")
        train_image_path=os.path.join(data_dir, "images", "train")
        # 导入数据
        train_df = pd.read_csv(train_csv_path)
        test_df = pd.read_csv(test_csv_path)

        # 分割验证集
        valid_df = train_df.sample(frac=0.1)

        # TFRecord文件路径
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # 构建tfrecord文件的完整路径
        valid_tfrecord = os.path.join(current_dir, 'valid.tfrecord')
        train_tfrecord = os.path.join(current_dir, 'train.tfrecord')
        test_tfrecord = os.path.join(current_dir, 'test.tfrecord')

        # 创建TFRecord文件
        create_tfrecord(train_image_path, valid_df, valid_tfrecord)
        create_tfrecord(train_image_path, train_df.drop(valid_df.index), train_tfrecord)
        create_tfrecord(test_image_path, test_df, test_tfrecord)

        # 解析TFRecord文件
        ds_train = tf.data.TFRecordDataset(train_tfrecord).map(_parse_function)
        ds_val = tf.data.TFRecordDataset(valid_tfrecord).map(_parse_function)
        ds_test = tf.data.TFRecordDataset(test_tfrecord).map(_parse_function)

        # 构建数据集信息
        ds_info = {
            'train_size': 400,
            'val_size': 40,
            'test_size': 103,
            # 其他信息
        }
        batch_size = 32  # 示例批处理大小
        caching = True  # 示例缓存设置
        return prepare(ds_train, ds_val, ds_test, ds_info, batch_size, caching)

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
    ds_train = ds_train.shuffle(371// 10)
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

