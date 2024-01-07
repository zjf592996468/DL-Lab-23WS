import numpy as np

def custom_confusion_matrix(y_true, y_pred, classes=2):
    matrix = np.zeros((classes, classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1
    return matrix

def custom_accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def custom_recall_score(y_true, y_pred, class_label):
    true_positive = np.sum((y_pred == class_label) & (y_true == class_label))
    false_negative = np.sum((y_pred != class_label) & (y_true == class_label))
    return true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0

def custom_auc_score(y_true, y_scores):
    # 这是一个复杂的计算，通常需要排序和插值。这里使用简化的方法。
    # 实际生产环境中推荐使用更复杂的实现。
    pos = y_scores[y_true == 1]
    neg = y_scores[y_true == 0]
    n_pos, n_neg = len(pos), len(neg)
    auc = 0.0
    for p in pos:
        auc += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return auc / (n_pos * n_neg) if n_pos * n_neg > 0 else 0

def custom_f1_score(y_true, y_pred, class_label):
    precision = custom_recall_score(y_true, y_pred, class_label)
    recall = custom_recall_score(y_true, y_pred, class_label)
    return 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
