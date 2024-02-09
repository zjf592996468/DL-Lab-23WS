import gin
import logging
import tensorflow as tf
from pathlib import Path
import pandas as pd
import numpy as np
from input_pipeline.preprocessing import slide_window, z_score, plot_df


def _bytes_feature(value):
    """TFRecord assist func: Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_tfrecord(dataset, tfrecord_path):
    """Creates a TFRecord to store datasets in tfrecord_path."""
    # Create a TFRecord writer
    with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
        # Iterate through the dataset and write each example to a TFRecord file
        for features, labels in dataset:
            feature_dict = {
                'features': _bytes_feature(tf.io.serialize_tensor(features)),
                'labels': _bytes_feature(tf.io.serialize_tensor(labels)),
            }

            example_proto = tf.train.Example(features=tf.train.Features(feature=feature_dict))
            writer.write(example_proto.SerializeToString())


def load_tfrd(tfrd_path, dataset_name):
    try:
        dataset = tf.data.TFRecordDataset(tfrd_path).map(parse_tfrecord_function)
    except FileNotFoundError as e:
        logging.warning(f"File not found for {dataset_name}: {tfrd_path}. Error: {e}")
        dataset = None

    return dataset


@gin.configurable()
def parse_tfrecord_function(example_proto, win_len):
    """Parse the example_proto from TensorFlow Examples"""
    # Define the feature
    feature_description = {
        'features': tf.io.FixedLenFeature([], tf.string),
        'labels': tf.io.FixedLenFeature([], tf.string),
    }

    # Parsing the input tf.train.Example proto using the defined feature dictionary
    example = tf.io.parse_single_example(example_proto, feature_description)

    # Decode the serialized features
    example['features'] = tf.io.parse_tensor(example['features'], out_type=tf.float64)
    example['features'] = tf.ensure_shape(example['features'], (win_len, 6))
    example['labels'] = tf.io.parse_tensor(example['labels'], out_type=tf.int32)
    example['labels'] = tf.ensure_shape(example['labels'], (win_len,))

    return example['features'], example['labels']


def df2win(df):
    """Transfer dataframe to tf.Dataset and cut it into windows"""
    features = df.drop(['label'], axis=1).values
    labels = df['label'].values
    dataset = tf.data.Dataset.from_tensor_slices((features.astype(np.float64), labels.astype(np.int32)))

    return slide_window(dataset)


@gin.configurable
def load(name, data_dir, show_exp_id):
    """Load the dataset"""
    if name == "hapt":
        logging.info(f"Preparing dataset {name}...")

        # Data directory path to read and store data
        rawdata_dir = Path(data_dir) / 'RawData'
        labels_path = Path(data_dir) / 'RawData' / 'labels.txt'
        act_labels_path = Path(data_dir) / 'activity_labels.txt'
        store_dir = Path.cwd().parent / 'results' / 'p2_HAR'
        store_dir.mkdir(parents=True, exist_ok=True)

        # Path to create TFRecord
        train_tfrd_path = store_dir / 'hapt_train.tfrecord'
        val_tfrd_path = store_dir / 'hapt_val.tfrecord'
        test_tfrd_path = store_dir / 'hapt_test.tfrecord'
        show_tfrd_path = store_dir / 'hapt_show.tfrecord'
        tfrd_paths = [train_tfrd_path, val_tfrd_path, test_tfrd_path, show_tfrd_path]

        # Create TFRecord when no TFRecord file exist and plot figure
        logging.info(f"Checking if TFRecord file exists in {store_dir}...")
        if not all(tfrd_path.is_file() for tfrd_path in tfrd_paths):
            logging.info(f"No TFRecord file found in {store_dir}, so creating.")

            # Load label data
            try:
                labels_df = pd.read_csv(labels_path, delim_whitespace=True, header=None,
                                        names=['exp_id', 'user_id', 'act_id', 'start', 'end'])
            except FileNotFoundError as e:
                logging.warning(f"File not found: {labels_path}. Error: {e}")

            # Initialize parameters to store the dataframe from different exp
            exp_dfs = {}
            current_exp_id = None
            current_user_id = None

            # Load acc and gyro data
            logging.info("Loading acc and gyro data...")
            for _, row in labels_df.iterrows():
                exp_id = row['exp_id']
                user_id = row['user_id']
                act_id = row['act_id'] - 1  # Use as integer labels
                start = row['start']
                end = row['end']

                # Check if exp_id or user_id has changed
                if exp_id != current_exp_id or user_id != current_user_id:
                    # Update current exp_id and user_id
                    current_exp_id = exp_id
                    current_user_id = user_id

                    # Signal files path
                    acc_data_path = rawdata_dir / f'acc_exp{exp_id:02d}_user{user_id:02d}.txt'
                    gyro_data_path = rawdata_dir / f'gyro_exp{exp_id:02d}_user{user_id:02d}.txt'

                    # Read acc and gyro data
                    try:
                        acc_data = pd.read_csv(acc_data_path, delim_whitespace=True, header=None,
                                               names=['acc_x', 'acc_y', 'acc_z'])
                    except FileNotFoundError as e:
                        logging.warning(f"File not found: {acc_data_path}. Error: {e}")

                    try:
                        gyro_data = pd.read_csv(gyro_data_path, delim_whitespace=True, header=None,
                                                names=['gyro_x', 'gyro_y', 'gyro_z'])
                    except FileNotFoundError as e:
                        logging.warning(f"File not found: {gyro_data_path}. Error: {e}")

                    # Combine acc and gyro data
                    sensor_data = pd.concat([acc_data, gyro_data], axis=1)

                    # Create -1 df of show_exp
                    if current_exp_id == show_exp_id:
                        plot_data = sensor_data.copy()
                        plot_data['label'] = -1

                # Label show_exp
                if current_exp_id == show_exp_id:
                    plot_data.loc[start:end, 'label'] = act_id

                # Extract labeled data from start point to end point
                sensor_subset = sensor_data.iloc[start:end, :].copy()
                sensor_subset['label'] = act_id

                # Create a new exp_df when exp changes
                if exp_id not in exp_dfs:
                    exp_dfs[exp_id] = pd.DataFrame(columns=sensor_subset.columns)

                # Combine labeled subsets in the same exp
                exp_dfs[exp_id] = pd.concat([exp_dfs[exp_id], sensor_subset], ignore_index=True)
            logging.info("Acc and gyro data with labels are extracted and combined.")

            # Plot raw data of show_exp
            fig = plot_df(plot_data, f'raw exp_{show_exp_id}')
            fig.savefig(store_dir / f'HAPT raw data of exp_{show_exp_id}.png')
            logging.info(f"HAPT raw data of exp_{show_exp_id} is saved to {store_dir.resolve()}")
            fig.close()

            # Plot show_exp with labeled data and save to file
            fig = plot_df(exp_dfs[show_exp_id], f'exp_{show_exp_id}')
            fig.savefig(store_dir / f'HAPT labeled data of exp_{show_exp_id}.png')
            logging.info(f"HAPT labeled data of exp_{show_exp_id} is saved to {store_dir.resolve()}")
            fig.close()

            # Z-score normalize
            exp_dfs_z = {}
            for exp_id in exp_dfs:
                exp_dfs_z[exp_id] = z_score(exp_dfs[exp_id].copy())
            logging.info("Sensor data with labels are z-score normalized on each channel.")

            # Plot show_exp after z-score and save to file
            fig = plot_df(exp_dfs_z[show_exp_id], f'exp_{show_exp_id} after z-score')
            fig.savefig(store_dir / f'HAPT labeled data of exp_{show_exp_id} after z-score.png')
            logging.info(f"HAPT labeled data of exp_{show_exp_id} after z-score is saved to {store_dir.resolve()}")
            fig.close()

            # Initialize train, val and test dataset
            ds_train = None
            ds_val = None
            ds_test = None

            # Get show dataset for evaluation
            features = exp_dfs_z[show_exp_id].drop(['label'], axis=1).values
            labels = exp_dfs_z[show_exp_id]['label'].values
            ds_show = tf.data.Dataset.from_tensor_slices((features.astype(np.float64), labels.astype(np.int32)))
            ds_show = ds_show.window(250, shift=250, stride=1, drop_remainder=True)  # Without oversample
            ds_show = (ds_show.flat_map(lambda feature, label: tf.data.Dataset.zip((feature, label)))
                       .batch(250, drop_remainder=True))

            # Window and build datasets
            for exp_id, df in exp_dfs_z.items():
                if 1 <= exp_id <= 43:
                    win_train = df2win(df)
                    if ds_train is None:
                        ds_train = win_train
                    else:
                        ds_train = ds_train.concatenate(win_train)
                elif 44 <= exp_id <= 55:
                    win_test = df2win(df)
                    if ds_test is None:
                        ds_test = win_test
                    else:
                        ds_test = ds_test.concatenate(win_test)
                else:
                    win_val = df2win(df)
                    if ds_val is None:
                        ds_val = win_val
                    else:
                        ds_val = ds_val.concatenate(win_val)
            logging.info("Dataset is seperated and windowed.")

            # Create TFRecord
            logging.info("Creating TFRecords...")
            create_tfrecord(ds_train, train_tfrd_path)
            create_tfrecord(ds_val, val_tfrd_path)
            create_tfrecord(ds_test, test_tfrd_path)
            create_tfrecord(ds_show, show_tfrd_path)
            logging.info(f"TFRecord files are created in {store_dir.resolve()}.")

        # Read TFRecord files and create dataset
        logging.info(f"Reading TFRecord files from {store_dir.resolve()}...")
        ds_train = load_tfrd(train_tfrd_path, 'train')
        ds_val = load_tfrd(val_tfrd_path, 'val')
        ds_test = load_tfrd(test_tfrd_path, 'test')
        ds_show = load_tfrd(show_tfrd_path, 'show')
        logging.info("Train, val, test and show datasets are created from TFRecords.")

        # # Check dataset labels
        # for parsed_record in ds_train.take(10):
        #     print(repr(parsed_record))

        # Initialize ds_info with num of acts and act names
        try:
            act_labels_df = pd.read_csv(act_labels_path, delim_whitespace=True, header=None,
                                        names=['act_id', 'act_name'])
        except FileNotFoundError as e:
            logging.warning(f"File not found: {act_labels_path}. Error: {e}")

        ds_info = {
            'num_acts': act_labels_df.shape[0],
            'act_names': act_labels_df['act_name'].tolist(),
        }

        # Calculate the size of dataset
        train_size = ds_train.reduce(tf.constant(0), lambda x, _: x + 1).numpy()
        val_size = ds_val.reduce(tf.constant(0), lambda x, _: x + 1).numpy()
        test_size = ds_test.reduce(tf.constant(0), lambda x, _: x + 1).numpy()
        ds_info.update({
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
        })
        logging.info(f"Num of train samples is: {train_size}")
        logging.info(f"Num of val samples is: {val_size}")
        logging.info(f"Num of test samples is: {test_size}")

        # Get the shape of features and labels
        ds_info.update({
            'features_shape': ds_train.element_spec[0].shape,
            'labels_shape': ds_train.element_spec[1].shape,
        })
        logging.info(f"ds_info is created: {ds_info}")

        return prepare(ds_train, ds_val, ds_test, ds_show, ds_info)


@gin.configurable
def prepare(ds_train, ds_val, ds_test, ds_show, ds_info, seed, batch_size, caching):
    """Prepare the dataset for training, validation and test"""
    # Prepare training dataset
    if caching:
        ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info['train_size'], seed=seed)  # shuffle whole dataset
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat(-1)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    # Prepare validation dataset
    ds_val = ds_val.batch(batch_size)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

    # Prepare test dataset
    ds_test = ds_test.batch(batch_size)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    # Prepare show dataset
    ds_show = ds_show.batch(batch_size)
    if caching:
        ds_show = ds_show.cache()
    ds_show = ds_show.prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_val, ds_test, ds_show, ds_info
