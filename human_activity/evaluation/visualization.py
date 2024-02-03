import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def visual(model, checkpoint, ds_test, ds_info):
    # load the trained model
    ckpt = tf.train.Checkpoint(model=model, optimizer=tf.keras.optimizers.Adam())
    ckpt.restore(checkpoint).expect_partial()

    # creat the list to get data
    true_labels = []
    pred_probs = []
    acc_data = []
    gyro_data = []

    # 循环遍历数据集并收集数据
    for x, y in ds_test:
        # Make predictions
        y_pred = model(x, training=False)

        # Add predictions and labels to the list
        true_labels.extend(y.numpy().flatten())  # Flatten labels
        pred_probs.extend(y_pred.numpy().flatten())  # Flatten prediction probability

        # get acc_data and gyro data
        acc_data.extend(x[:, :, 0:3].numpy().flatten())
        gyro_data.extend(x[:, :, 3:6].numpy().flatten())

    # 将列表转换为NumPy数组
    acc_data = np.array(acc_data).reshape(-1, 3)
    gyro_data = np.array(gyro_data).reshape(-1, 3)

    # Convert list to NumPy array
    true_labels = np.array(true_labels)
    pred_probs = np.array(pred_probs)

    # Get the index of maximum probability as prediction category based on prediction probability
    pred_labels = np.argmax(pred_probs.reshape(-1, ds_info['num_acts']), axis=1)

    # colr mapping
    unique_labels = np.unique(true_labels)
    colors_pred = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    # get length of data
    data_indices = np.arange(len(acc_data))

    # plot the predict and true label
    acc_labels = ['ACC X', 'ACC Y', 'ACC Z']
    gyro_labels = ['GYRO X', 'GYRO Y', 'GYRO Z']
    ig, axs = plt.subplots(4, 1)

    # plot 1 'Accelerometer Data with True Label Background'
    for i, label in enumerate(true_labels):
        label_index = np.where(unique_labels == label)[0][0]
        color = colors_pred[label_index]
        axs[0].axvspan(i, i + 1, color=color, alpha=0.1)
    for i in range(3):
        axs[0].plot(data_indices, acc_data[:, i], label=acc_labels[i])
    axs[0].legend()
    axs[0].set_title('Accelerometer Data with True Label Background')

    # plot 2
    for i, label in enumerate(pred_labels):
        label_index = np.where(unique_labels == label)[0][0]
        color = colors_pred[label_index]
        axs[1].axvspan(i, i + 1, color=color, alpha=0.1)
    for i in range(3):
        axs[1].plot(data_indices, acc_data[:, i], label=gyro_labels[i])
    axs[1].legend()
    axs[1].set_title('Accelerometer Data with Predicted Label Background')

    # 绘制加速度计数据与预测标签背景
    for i, label in enumerate(true_labels):
        label_index = np.where(unique_labels == label)[0][0]
        color = colors_pred[label_index]
        axs[2].axvspan(i, i + 1, color=color, alpha=0.1)
    for i in range(3):
        axs[2].plot(data_indices, gyro_data[:, i], label=acc_labels[i])
    axs[2].legend()
    axs[2].set_title('Gyroscope Data with True Label Background')

    # 绘制陀螺仪数据与预测标签背景
    for i, label in enumerate(pred_labels):
        label_index = np.where(unique_labels == label)[0][0]
        color = colors_pred[label_index]
        axs[3].axvspan(i, i + 1, color=color, alpha=0.1)
    for i in range(3):
        axs[3].plot(data_indices, gyro_data[:, i], label=gyro_labels[i])
    axs[3].legend()
    axs[3].set_title('Gyroscope Data with Predicted Label Background')

    action_names = ds_info['act_names']
    num_actions = len(action_names)
    # Create a color map for these actions
    colors_actions = plt.cm.rainbow(np.linspace(0, 1, num_actions))

    # Creating the legend figure with adjusted size
    fig, ax = plt.subplots(figsize=(15, 3))  # Adjust figure size if needed
    for i, color in enumerate(colors_actions):
        ax.fill_between([i, i + 1], 0, 1, color=color, alpha=0.5)  # Fill the entire horizontal space

    # Set the x-axis to span the width of the color bands
    ax.set_xlim(0, num_actions)
    ax.set_ylim(0, 1)

    # Remove y-axis as it's not needed
    ax.set_yticks([])

    # Set the x-axis ticks and labels
    ax.set_xticks(np.arange(0.5, num_actions, 1))
    ax.set_xticklabels(action_names, rotation=90, ha='center')  # ha='center' centers the labels

    # Ensure the entire plot fits into the figure area
    plt.tight_layout()

    # Add a title and show the plot
    plt.title("Action Classes Legend")

    plt.show()
