import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import wandb
from evaluation.metrics import custom_recall_score, custom_auc_score, custom_f1_score, custom_accuracy_score, \
    custom_confusion_matrix
from sklearn.metrics import roc_auc_score


def evaluate(model: tf.keras.Model, checkpoint: object, ds_test: tf.data.Dataset, run_paths: dict) -> np.ndarray:
    ckpt = tf.train.Checkpoint(model=model, optimizer=tf.keras.optimizers.Adam())
    ckpt.restore(checkpoint).expect_partial()
    wandb.init(project='idrid', name=run_paths['model_id'])

    true_labels = []
    pred_probs = []

    for x, y in ds_test:
        y_pred = model(x, training=False)  # 直接在模型上调用 x
        true_labels.extend(y.numpy())
        pred_probs.extend(y_pred.numpy())  # 使用 numpy() 转换

    true_labels = np.array(true_labels)
    pred_labels = np.argmax(pred_probs, axis=1)

    # 使用自定义函数计算指标
    conf_matrix = custom_confusion_matrix(true_labels, pred_labels)
    accuracy = custom_accuracy_score(true_labels, pred_labels)
    sensitivity = custom_recall_score(true_labels, pred_labels, 1)
    specificity = custom_recall_score(true_labels, pred_labels, 0)
    auc = roc_auc_score(true_labels, [pred[1] for pred in pred_probs])
    f1_score = custom_f1_score(true_labels, pred_labels, 1)

    # Log metrics
    logging.info("Confusion Matrix:\n%s", conf_matrix)
    logging.info("Accuracy: %s", accuracy)
    logging.info("Sensitivity: %s", sensitivity)
    logging.info("Specificity: %s", specificity)
    logging.info("ROC/AUC: %s", auc)
    logging.info("f1_score:%s", f1_score)

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


def evaluate1(model: tf.keras.Model, ds_test: tf.data.Dataset, run_paths) -> np.ndarray:
    # Initialize Weights & Biases
    wandb.init(project='idrid', name=run_paths['model_id'])

    true_labels = []
    pred_probs = []

    for x, y in ds_test:
        y_pred = model(x, training=False)  # 直接在模型上调用 x
        true_labels.extend(y.numpy())
        pred_probs.extend(y_pred.numpy())  # 使用 numpy() 转换

    true_labels = np.array(true_labels)
    pred_labels = np.argmax(pred_probs, axis=1)

    # 使用自定义函数计算指标
    conf_matrix = custom_confusion_matrix(true_labels, pred_labels)
    accuracy = custom_accuracy_score(true_labels, pred_labels)
    sensitivity = custom_recall_score(true_labels, pred_labels, 1)
    specificity = custom_recall_score(true_labels, pred_labels, 0)
    auc = roc_auc_score(true_labels, [pred[1] for pred in pred_probs])
    f1_score = custom_f1_score(true_labels, pred_labels, 1)

    # Log metrics
    logging.info("Confusion Matrix:\n%s", conf_matrix)
    logging.info("Accuracy: %s", accuracy)
    logging.info("Sensitivity: %s", sensitivity)
    logging.info("Specificity: %s", specificity)
    logging.info("ROC/AUC: %s", auc)
    logging.info("f1_score:%s", f1_score)

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
