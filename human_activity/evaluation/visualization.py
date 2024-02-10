import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
import logging
import wandb
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

    # get feature and label
    for x, y in ds_test:
        # Make predictions
        y_pred = model(x, training=False)

        # Add predictions and labels to the list
        true_labels.extend(y.numpy().flatten())  # Flatten labels
        pred_probs.extend(y_pred.numpy().flatten())  # Flatten prediction probability

        # get acc_data and gyro data
        acc_data.extend(x[:, :, 0:3].numpy().flatten())
        gyro_data.extend(x[:, :, 3:6].numpy().flatten())

    # list to array
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
    for i, color in enumerate(colors_pred):
        colors_pred[i, :] = to_rgba(color, alpha=0.5)

    # get length of data
    data_indices = np.arange(len(acc_data))

    # plot the predict and true label
    acc_labels = ['ACC X', 'ACC Y', 'ACC Z']
    gyro_labels = ['GYRO X', 'GYRO Y', 'GYRO Z']

    # Create subplots using the GridSpec object
    fig, axs = plt.subplots(5, 1, figsize=(16, 13), gridspec_kw={'height_ratios': [1, 1, 1, 1, 0.3]})

    # plot 1 'Accelerometer Data with True Label Background'
    for i, label in enumerate(true_labels):
        label_index = np.where(unique_labels == label)[0][0]
        color = colors_pred[label_index]
        axs[0].axvspan(i, i + 1, color=color, linewidth=0)
    for i in range(3):
        axs[0].plot(data_indices, acc_data[:, i], label=acc_labels[i])
    axs[0].legend()
    axs[0].set_title('Accelerometer Data with True Label Background')

    # plot 2
    for i, label in enumerate(pred_labels):
        label_index = np.where(unique_labels == label)[0][0]
        color = colors_pred[label_index]
        axs[1].axvspan(i, i + 1, color=color, linewidth=0)
    for i in range(3):
        axs[1].plot(data_indices, acc_data[:, i], label=gyro_labels[i])
    axs[1].legend()
    axs[1].set_title('Accelerometer Data with Predicted Label Background')

    # plot 3 color and data
    for i, label in enumerate(true_labels):
        label_index = np.where(unique_labels == label)[0][0]
        color = colors_pred[label_index]
        axs[2].axvspan(i, i + 1, color=color, linewidth=0)
    for i in range(3):
        axs[2].plot(data_indices, gyro_data[:, i], label=acc_labels[i])
    axs[2].legend()
    axs[2].set_title('Gyroscope Data with True Label Background')

    # plot 4
    for i, label in enumerate(pred_labels):
        label_index = np.where(unique_labels == label)[0][0]
        color = colors_pred[label_index]
        axs[3].axvspan(i, i + 1, color=color, linewidth=0)
    for i in range(3):
        axs[3].plot(data_indices, gyro_data[:, i], label=gyro_labels[i])
    axs[3].legend()
    axs[3].set_title('Gyroscope Data with Predicted Label Background')

    action_names = ds_info['act_names']
    num_actions = len(action_names)

    # Create a color map for these actions
    colors_actions = plt.cm.rainbow(np.linspace(0, 1, num_actions))

    for i, color in enumerate(colors_actions):
        axs[4].fill_between([i, i + 1], 0, 1, color=color, alpha=0.5, linewidth=0)  # Fill the entire horizontal space

    # Set the x-axis to span the width of the color bands
    axs[4].set_xlim(0, num_actions)
    axs[4].set_ylim(0, 1)

    # Remove y-axis as it's not needed
    axs[4].set_yticks([])

    # Set the x-axis ticks and labels
    axs[4].set_xticks(np.arange(0.5, num_actions, 1))
    axs[4].set_xticklabels(action_names, rotation=45, ha='center')  # ha='center' centers the labels

    # Ensure the entire plot fits into the figure area
    plt.tight_layout()

    # Add a title and show the plot
    plt.title("Action Classes Legend")

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(true_labels, pred_labels)

    # Convert confusion matrix to percentage
    conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    accuracy = accuracy_score(true_labels, pred_labels)
    balanced_accuracy = balanced_accuracy_score(true_labels, pred_labels)
    logging.info("Confusion Matrix:\n%s", conf_matrix)
    logging.info("Accuracy: %s", accuracy)
    logging.info("balanced Accuracy: %s", balanced_accuracy)

    # Use wandb to record confusion matrix and accuracy
    wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(probs=None, y_true=true_labels, preds=pred_labels,
                                                               class_names=ds_info['act_names']),
               "accuracy": accuracy,
               "balanced Accuracy": balanced_accuracy})

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt=".2%", cmap='Blues',
                xticklabels=ds_info['act_names'], yticklabels=ds_info['act_names'])

    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    return conf_matrix
