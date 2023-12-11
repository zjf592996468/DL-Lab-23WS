from evaluation.metrics import ConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import numpy as np

def evaluate(model: tf.keras.Model, checkpoint:object, ds_test: tf.data.Dataset, run_paths: dict) -> np.ndarray:
    ckpt = tf.train.Checkpoint(model=model, optimizer=tf.keras.optimizers.Adam())
    manager = tf.train.CheckpointManager(ckpt, run_paths['path_ckpts_train'], max_to_keep=3)
    ckpt_restore_path = manager.latest_checkpoint
    ckpt.restore(ckpt_restore_path).expect_partial()

    # 初始化混淆矩阵度量
    confusion_matrix_metric = ConfusionMatrix(num_classes=2)

    # 在测试数据集上评估模型
    for x, y in ds_test:
        y_pred = model.predict(x)
        confusion_matrix_metric.update_state(y, y_pred)

    # 获取混淆矩阵
    conf_matrix = confusion_matrix_metric.result().numpy()

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()
    print("Confusion Matrix:")
    print(conf_matrix)
    return conf_matrix

def evaluate1(model: tf.keras.Model, ds_test: tf.data.Dataset) -> np.ndarray:


    # 初始化混淆矩阵度量
    confusion_matrix_metric = ConfusionMatrix(num_classes=2)

    # 在测试数据集上评估模型
    for x, y in ds_test:
        y_pred = model.predict(x)
        confusion_matrix_metric.update_state(y, y_pred)

    # 获取混淆矩阵
    conf_matrix = confusion_matrix_metric.result().numpy()

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()
    print("Confusion Matrix:")
    print(conf_matrix)
    return conf_matrix