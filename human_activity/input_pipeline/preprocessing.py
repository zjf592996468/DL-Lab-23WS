import gin
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
from itertools import groupby


def z_score(dataframe):
    """Normalize signals on each channel independently with Z-Score normalization"""
    channels = dataframe.columns.difference(['label'])
    for channel in channels:
        mean = dataframe[channel].mean()
        std = dataframe[channel].std()
        dataframe[channel] = (dataframe[channel] - mean) / std

    return dataframe


@gin.configurable()
def slide_window(dataset, win_len, win_shift):
    """Employ the sliding window technique"""
    dataset = dataset.window(win_len, shift=win_shift, stride=1, drop_remainder=True)

    # Flatten the windowed dataset
    dataset = (dataset.flat_map(lambda feature, label: tf.data.Dataset.zip((feature, label)))
               .batch(win_len, drop_remainder=True))

    return dataset


def plot_df(dataframe, title):
    """Plot the dataframe"""
    columns_acc = ['acc_x', 'acc_y', 'acc_z']
    columns_gyro = ['gyro_x', 'gyro_y', 'gyro_z']

    # Create a figure with two subplots (2 rows, 1 column)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))  # option: sharex=True

    # Check whether dataframe is dataframe
    if isinstance(dataframe, pd.DataFrame):
        # Get label column
        label_column = np.sort(dataframe['label'].unique())
        label_column = label_column[label_column != -1]

        for label in label_column:
            conditions = (dataframe['label'] == label)

            # Find multiple start and end points
            sorted_indices = sorted(dataframe.index[conditions].tolist())
            grouped_indices = [list(group) for _, group in
                               groupby(enumerate(sorted_indices), key=lambda x: x[0] - x[1])]

            # Plot labels as background
            for group in grouped_indices:
                start_index = group[0][1]
                end_index = group[-1][1]

                # Get rainbow color for each label
                rgba_color = to_rgba(plt.cm.rainbow(label / (len(label_column) - 1)))
                ax1.axvspan(start_index, end_index + 1, color=rgba_color, alpha=0.5, linewidth=0)
                ax2.axvspan(start_index, end_index + 1, color=rgba_color, alpha=0.5, linewidth=0)

        for column in columns_acc:
            ax1.plot(dataframe[column], label=column)
        for column in columns_gyro:
            ax2.plot(dataframe[column], label=column)

    else:
        features, labels = dataframe
        for column, values in zip(columns_acc, tf.unstack(features, axis=-1)):
            ax1.plot(values, label=column)
        for column, values in zip(columns_gyro, tf.unstack(features, axis=-1)):
            ax2.plot(values, label=column)

    # Plot accelerometer data
    ax1.set_ylabel('Accelerometer Value')
    ax1.set_title(f'Accelerometer Data of {title}')
    ax1.legend()

    # Plot gyroscope data
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Gyroscope Value')
    ax2.set_title(f'Gyroscope Data of {title}')
    ax2.legend()

    plt.tight_layout()  # Adjust layout for better spacing

    return plt
