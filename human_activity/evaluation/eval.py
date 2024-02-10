import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
import logging
import wandb
import tensorflow as tf


def log_and_plot_metrics(true_labels, pred_labels, ds_info, use_wandb=True):
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(true_labels, pred_labels)

    # Convert confusion matrix to percentage
    conf_matrix_percentage = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    #  Calculate accuracy and balanced_accuracy
    accuracy = accuracy_score(true_labels, pred_labels)
    balanced_accuracy = balanced_accuracy_score(true_labels, pred_labels)

    # login
    logging.info("Confusion Matrix:\n%s", conf_matrix_percentage)
    logging.info("Accuracy: %s", accuracy)
    logging.info("Balanced Accuracy: %s", balanced_accuracy)

    # Use wandb to record confusion matrix and accuracy
    if use_wandb:
        wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(probs=None, y_true=true_labels, preds=pred_labels,
                                                                   class_names=ds_info['act_names']),
                   "accuracy": accuracy,
                   "balanced_accuracy": balanced_accuracy})

    # plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix_percentage, annot=True, fmt=".2%", cmap='Blues',
                xticklabels=ds_info['act_names'], yticklabels=ds_info['act_names'])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    return conf_matrix


def evaluate(model, checkpoint, ds_test, ds_info):
    ckpt = tf.train.Checkpoint(model=model, optimizer=tf.keras.optimizers.Adam())
    ckpt.restore(checkpoint).expect_partial()

    true_labels = []
    pred_probs = []

    # Iterate over the test dataset
    for x, y in ds_test:
        # Make predictions
        y_pred = model(x, training=False)

        # Add predictions and labels to the list
        true_labels.extend(y.numpy().flatten())  # Flatten labels
        pred_probs.extend(y_pred.numpy().flatten())  # Flatten prediction probability

    # Convert list to NumPy array
    true_labels = np.array(true_labels)
    pred_probs = np.array(pred_probs)

    # Get the index of maximum probability as prediction category based on prediction probability
    pred_labels = np.argmax(pred_probs.reshape(-1, ds_info['num_acts']), axis=1)

    # Plot confusion matrix and log into wandb
    conf_matrix = log_and_plot_metrics(true_labels, pred_labels, ds_info, True)

    return conf_matrix
