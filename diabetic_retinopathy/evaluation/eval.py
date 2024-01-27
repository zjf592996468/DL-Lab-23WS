import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import wandb
from evaluation.metrics import recall_score, auc_score, f1score, accuracy_score, confusion_matrix
from absl.flags import FLAGS


def evaluate(model: tf.keras.Model, checkpoint: object, ds_test: tf.data.Dataset, ds_info) -> np.ndarray:
    ckpt = tf.train.Checkpoint(model=model, optimizer=tf.keras.optimizers.Adam())
    ckpt.restore(checkpoint).expect_partial()

    # make one empty dict to get value
    true_labels = []
    pred_probs = []

    # put the value into empty dict
    for x, y in ds_test:
        y_pred = model(x, training=False)  # Calling x directly on the model
        true_labels.extend(y.numpy())
        pred_probs.extend(y_pred.numpy())  # Conversion with numpy()

    true_labels = np.array(true_labels)
    if FLAGS.multi_class:
        pred_labels = np.clip(np.round(pred_probs), 0, 4).astype(int).squeeze()
    else:
        pred_labels = np.argmax(pred_probs, axis=1)

    # Calculating metrics using custom functions
    conf_matrix = confusion_matrix(true_labels, pred_labels, ds_info['num_classes'])
    accuracy = accuracy_score(true_labels, pred_labels)

    # login and wandb login
    logging.info("Confusion Matrix:\n%s", conf_matrix)
    logging.info("Accuracy: %s", accuracy)
    wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(probs=None, y_true=true_labels,
                                                               preds=pred_labels,
                                                               class_names=[f"Class {i}" for i in
                                                                            range(ds_info['num_classes'])]),
               "accuracy": accuracy})

    # Use flags to control the evaluation
    if not FLAGS.multi_class:
        sensitivity = recall_score(true_labels, pred_labels, 1)
        specificity = recall_score(true_labels, pred_labels, 0)
        auc = auc_score(true_labels, [pred[1] for pred in pred_probs])
        f1_score = f1score(true_labels, pred_labels, 1)
        logging.info("Sensitivity: %s", sensitivity)
        logging.info("Specificity: %s", specificity)
        logging.info("ROC/AUC: %s", auc)
        logging.info("f1_score:%s", f1_score)
        wandb.log({"sensitivity": sensitivity, "specificity": specificity, "roc_auc": auc})

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

    # Ensure wandb run finishes
    wandb.finish()

    return conf_matrix


def evaluate1(model: tf.keras.Model, ds_test: tf.data.Dataset, ds_info) -> np.ndarray:
    # this function can directly evaluate the model

    # make one empty dict to get value
    true_labels = []
    pred_probs = []

    # put the value into empty dict
    for x, y in ds_test:
        y_pred = model(x, training=False)  # Calling x directly on the model
        true_labels.extend(y.numpy())
        pred_probs.extend(y_pred.numpy())  # Conversion with numpy()

    true_labels = np.array(true_labels)
    if FLAGS.multi_class:
        pred_labels = np.clip(np.round(pred_probs), 0, 4).astype(int).squeeze()
    else:
        pred_labels = np.argmax(pred_probs, axis=1)

    # Calculating metrics using custom functions
    conf_matrix = confusion_matrix(true_labels, pred_labels, ds_info['num_classes'])
    accuracy = accuracy_score(true_labels, pred_labels)

    # login and wandb login
    logging.info("Confusion Matrix:\n%s", conf_matrix)
    logging.info("Accuracy: %s", accuracy)
    wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(probs=None, y_true=true_labels,
                                                               preds=pred_labels,
                                                               class_names=[f"Class {i}" for i in
                                                                            range(ds_info['num_classes'])]),
               "accuracy": accuracy})

    # Use flags to control the evaluation
    if not FLAGS.multi_class:
        sensitivity = recall_score(true_labels, pred_labels, 1)
        specificity = recall_score(true_labels, pred_labels, 0)
        auc = auc_score(true_labels, [pred[1] for pred in pred_probs])
        f1_score = f1score(true_labels, pred_labels, 1)
        logging.info("Sensitivity: %s", sensitivity)
        logging.info("Specificity: %s", specificity)
        logging.info("ROC/AUC: %s", auc)
        logging.info("f1_score:%s", f1_score)
        wandb.log({"sensitivity": sensitivity, "specificity": specificity, "roc_auc": auc})

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

    # Ensure wandb run finishes
    wandb.finish()

    return conf_matrix
