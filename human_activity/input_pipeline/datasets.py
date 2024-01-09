import gin
import logging
import tensorflow as tf
from pathlib import Path


@gin.configurable
def load(name, data_dir, split_frac, seed):
    """Load the dataset"""
    if name == "hapt":
        logging.info(f"Preparing dataset {name}...")

        # Data directory path to read and store data
        hapt_dir = Path(data_dir) / 'RawData'
        labels_path = Path(data_dir) / 'RawData' / 'labels.txt'
        store_dir = Path.cwd().parent.parent

        # Path to create TFRecord
        train_tfrd_path = store_dir / 'har_train.tfrecord'
        val_tfrd_path = store_dir / 'har_val.tfrecord'
        test_tfrd_path = store_dir / 'har_test.tfrecord'

        # todo: load sensors' signal
    return
