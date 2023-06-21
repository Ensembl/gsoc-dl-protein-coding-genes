import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, PrecisionRecallDisplay, roc_curve, RocCurveDisplay
import numpy as np

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Gene', 'Gene'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path, dpi=900)
    plt.show()

def plot_precision_recall_curve(y_true, probabilities, save_path=None):
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()
    plt.title('Precision-Recall Curve')
    if save_path:
        plt.savefig(save_path, dpi=900)
    plt.show()

def plot_roc_curve(y_true, probabilities, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    disp = RocCurveDisplay(fpr=fpr, tpr=tpr)
    disp.plot()
    plt.title('ROC Curve')
    if save_path:
        plt.savefig(save_path, dpi=900)
    plt.show()

def plot_histogram(values, title, save_path=None):
    plt.hist(values, bins=20)
    plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=900)
    plt.show()

def plot_sequence_probabilities(true_labels, predicted_probabilities, save_path=None):
    fig, ax = plt.subplots(figsize=(20, 6))
    tokens = np.arange(len(true_labels))

    ax.plot(tokens, predicted_probabilities, marker='o', linestyle='-', label='Predicted Probability')
    ax.scatter(tokens, true_labels, marker='o', color='red', label='True Label')

    ax.set_xlabel('Token Index')
    ax.set_ylabel('Probability / Label')
    ax.set_title('Sequence Probabilities')

    ax.legend()
    if save_path:
        plt.savefig(save_path, dpi=900)
    plt.show()
