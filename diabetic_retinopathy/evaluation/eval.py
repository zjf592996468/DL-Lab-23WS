from evaluation.metrics import ConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(model: object, checkpoint: object, ds_test: object) -> object:
    model.load_weights(checkpoint)

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

def evaluate1(model: object, checkpoint: object, ds_test: object) -> object:


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