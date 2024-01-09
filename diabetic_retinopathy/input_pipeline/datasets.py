import gin
import logging
import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path
import pandas as pd
from absl.flags import FLAGS
from input_pipeline.preprocessing import preprocess, augment, check_imb


def _bytes_feature(value):
    """TFRecord assist func: Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """TFRecord assist func: Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_tfrecord(tfrd_path, img_dir, labels):
    """Create a TFRecord at 'tfrd_path' with datas from 'img_dir' and 'labels'"""
    with tf.io.TFRecordWriter(str(tfrd_path)) as writer:
        # According to labels to read files
        for index, row in labels.iterrows():
            try:
                img_path = img_dir / (row['Image name'] + '.jpg')
                img = open(img_path, 'rb').read()
                label = row['Retinopathy grade']

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
    # Defining features
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    # Parsing features from proto
    example = tf.io.parse_single_example(example_proto, feature_description)

    # Decoding JPEG image data
    example['image'] = tf.io.decode_jpeg(example['image'], channels=3)

    return example['image'], example['label']


@gin.configurable
def load(name, data_dir, split_frac, seed):
    """Load the dataset"""
    if name == "idrid":
        logging.info(f"Preparing dataset {name}...")

        # Data directory path to read and store data
        train_img_dir = Path(data_dir) / 'images' / 'train'
        test_img_dir = Path(data_dir) / 'images' / 'test'
        labels_dir = Path(data_dir) / 'labels'
        store_dir = Path.cwd().parent.parent

        # Path to create TFRecord
        train_tfrd_path = store_dir / 'train.tfrecord'
        val_tfrd_path = store_dir / 'val.tfrecord'
        test_tfrd_path = store_dir / 'test.tfrecord'

        # Read label files, only read rows of "Image name" and "Retinopathy grade"
        train_labels = pd.read_csv(Path(labels_dir) / 'train.csv', usecols=['Image name', 'Retinopathy grade'])
        train_labels = train_labels  # Drop duplicate data
        test_labels = pd.read_csv(Path(labels_dir) / 'test.csv', usecols=['Image name', 'Retinopathy grade'])

        # Check and plot the distribution of raw dataset
        fig = check_imb(train_labels)
        fig.xlabel('Retinopathy grade')
        fig.title('Class Distribution of IDRID dataset')
        fig.savefig(store_dir / 'Class distribution of IDRID dataset.png')
        logging.info(f"Class distribution of IDRID dataset is saved to {store_dir.resolve()}")
        fig.close()

        # Split train and val dataset with split_frac
        # # First random shuffle the train dataset
        # train_labels = train_labels.sample(frac=1, random_state=seed).reset_index(drop=True)
        # Without shuffle
        val_size = int(split_frac * train_labels.shape[0])
        train_size = train_labels.shape[0] - val_size
        val_dataset = train_labels[train_size:]
        train_dataset = train_labels[:train_size]
        logging.info(f"Dataset is divided into train and validation with rate = {split_frac}.")
        logging.info(f"Num of train samples before resampling is: {train_size}")
        logging.info(f"Num of val samples is: {val_size}")
        logging.info(f"Num of test samples is: {test_labels.shape[0]}")

        # Binarise the dataset when not doing multi classification
        if not FLAGS.multi_class:
            # Group all data into 2 groups, with 0 represents NRDR, 1 represents RDR
            train_dataset['Retinopathy grade'] = (train_dataset['Retinopathy grade'] > 1).astype(int)
            val_dataset['Retinopathy grade'] = (val_dataset['Retinopathy grade'] > 1).astype(int)
            test_labels['Retinopathy grade'] = (test_labels['Retinopathy grade'] > 1).astype(int)

            # Check and plot the distribution of binarised dataset
            fig = check_imb(train_dataset)
            fig.title('Class Distribution Of Train Set After Binarisation')
            fig.savefig(store_dir / 'Class distribution of train set after binarisation.png')
            logging.info(f"Class distribution of train set after binarisation is saved to {store_dir.resolve()}")
            fig.close()

        # Resample the train dataset with oversampling
        class_counts = train_dataset['Retinopathy grade'].value_counts().sort_index()
        targ_size = class_counts.max()
        train_dataset = train_dataset.groupby('Retinopathy grade').apply(
            lambda x: x.sample(targ_size, replace=True, random_state=seed))
        train_dataset = train_dataset.sample(frac=1, random_state=seed).reset_index(drop=True)
        logging.info(f"Train dataset is resampled.")
        train_size = train_dataset.shape[0]
        logging.info(f"Num of train samples after resampling is: {train_size}")

        # Check and plot the distribution of resampled dataset
        fig = check_imb(train_dataset)
        fig.title('Class Distribution Of Train Set After Resampling')
        fig.savefig(store_dir / 'Class distribution of train set after resampling.png')
        logging.info(f"Class distribution of train set after resampling is saved to {store_dir.resolve()}")
        fig.close()

        # Build dataset info
        class_counts = train_dataset['Retinopathy grade'].value_counts().sort_index()
        ds_info = {
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_labels.shape[0],
            'num_classes': class_counts.shape[0],
            'class0_counts': class_counts.values[0],
            'class1_counts': class_counts.values[1],
        }

        # Update ds_info when doing multi classification
        if FLAGS.multi_class:
            ds_info.update({
                'class2_counts': class_counts.values[2],
                'class3_counts': class_counts.values[3],
                'class4_counts': class_counts.values[4],
            })

        # Create TFRecord files for train, val and test
        create_tfrecord(train_tfrd_path, train_img_dir, train_dataset)
        create_tfrecord(val_tfrd_path, train_img_dir, val_dataset)
        create_tfrecord(test_tfrd_path, test_img_dir, test_labels)
        logging.info(f"TFRecord files are created in {store_dir.resolve()}.")

        # Read TFRecord files and create dataset
        ds_train = tf.data.TFRecordDataset(train_tfrd_path).map(_parse_tfrd_function)
        ds_val = tf.data.TFRecordDataset(val_tfrd_path).map(_parse_tfrd_function)
        ds_test = tf.data.TFRecordDataset(test_tfrd_path).map(_parse_tfrd_function)
        logging.info("Train, val and test datasets are created from tfrecord.")

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
def prepare(ds_train, ds_val, ds_test, ds_info, seed, batch_size, caching):
    """Prepare the dataset for training, validation and test"""
    # Prepare training dataset
    ds_train = ds_train.map(
        preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    # get image shape and update ds_info
    for image, label in ds_train.take(1):
        shape = image.numpy().shape
    ds_info.update({
        'shape': shape,
        'image_height': shape[0],
        'image_width': shape[1],
        'image_channels': shape[2],
    })
    logging.info('Dataset info ds_info is built.')

    if caching:
        ds_train = ds_train.cache()
    ds_train = ds_train.map(
        augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.shuffle(ds_info['train_size'], seed=seed)  # shuffle whole dataset
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat(-1)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    # Prepare validation dataset
    ds_val = ds_val.map(
        preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds_val = ds_val.batch(batch_size)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

    # Prepare test dataset
    ds_test = ds_test.map(
        preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_val, ds_test, ds_info
