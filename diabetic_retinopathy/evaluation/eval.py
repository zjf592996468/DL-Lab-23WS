import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import wandb
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, recall_score

def evaluate(model: tf.keras.Model, checkpoint:object, ds_test: tf.data.Dataset, run_paths: dict) -> np.ndarray:
    ckpt = tf.train.Checkpoint(model=model, optimizer=tf.keras.optimizers.Adam())
    manager = tf.train.CheckpointManager(ckpt, run_paths['path_ckpts_train'], max_to_keep=3)
    ckpt_restore_path = manager.latest_checkpoint
    ckpt.restore(ckpt_restore_path).expect_partial()
    wandb.init(project='idrid', name=run_paths['model_id'])

    true_labels = []
    pred_probs = []

    for x, y in ds_test:
        y_pred = model.predict(x)
        true_labels.extend(y.numpy())
        pred_probs.extend(y_pred)

    true_labels = np.array(true_labels)
    pred_labels = np.argmax(pred_probs, axis=1)

    # Calculate metrics
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    accuracy = accuracy_score(true_labels, pred_labels)
    sensitivity = recall_score(true_labels, pred_labels)
    specificity = recall_score(true_labels, pred_labels, pos_label=0)
    auc = roc_auc_score(true_labels, [pred[1] for pred in pred_probs])

    # Log metrics
    logging.info("Confusion Matrix:\n%s", conf_matrix)
    logging.info("Accuracy: %s", accuracy)
    logging.info("Sensitivity: %s", sensitivity)
    logging.info("Specificity: %s", specificity)
    logging.info("ROC/AUC: %s", auc)

    wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(probs=None, y_true=true_labels, preds=pred_labels,
                                                               class_names=["Class 0", "Class 1"]),
               "accuracy": accuracy,
               "sensitivity": sensitivity,
               "specificity": specificity,
               "roc_auc": auc})

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


def evaluate1(model: tf.keras.Model, ds_test: tf.data.Dataset, run_name: str) -> np.ndarray:
    # Initialize Weights & Biases
    wandb.init(project='idrid', name=run_paths['model_id'])

    true_labels = []
    pred_probs = []

    for x, y in ds_test:
        y_pred = model.predict(x)
        true_labels.extend(y.numpy())
        pred_probs.extend(y_pred)

    true_labels = np.array(true_labels)
    pred_labels = np.argmax(pred_probs, axis=1)

    # Calculate metrics
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    accuracy = accuracy_score(true_labels, pred_labels)
    sensitivity = recall_score(true_labels, pred_labels)
    specificity = recall_score(true_labels, pred_labels, pos_label=0)
    auc = roc_auc_score(true_labels, [pred[1] for pred in pred_probs])

    # Log metrics
    logging.info("Confusion Matrix:\n%s", conf_matrix)
    logging.info("Accuracy: %s", accuracy)
    logging.info("Sensitivity: %s", sensitivity)
    logging.info("Specificity: %s", specificity)
    logging.info("ROC/AUC: %s", auc)

    wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(probs=None, y_true=true_labels, preds=pred_labels,
                                                               class_names=["Class 0", "Class 1"]),
               "accuracy": accuracy,
               "sensitivity": sensitivity,
               "specificity": specificity,
               "roc_auc": auc})

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