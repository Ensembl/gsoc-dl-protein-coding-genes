import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, PrecisionRecallDisplay, roc_curve, RocCurveDisplay
import numpy as np


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=['Not Gene', 'Gene'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path, dpi=900)
    plt.show()


def plot_precision_recall_curve(y_true, probabilities, save_path=None):
    precision, recall, _ = precision_recall_curve(
        y_true, probabilities, pos_label='gene')
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()
    plt.title('Precision-Recall Curve')
    if save_path:
        plt.savefig(save_path, dpi=900)
    plt.show()


def plot_roc_curve(y_true, probabilities, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, probabilities, pos_label='gene')
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


def plot_sequence_labels(true_labels, predicted_labels, save_path=None):
    fig, ax = plt.subplots(figsize=(20, 6))
    tokens = np.arange(len(true_labels))




    # Plot predicted labels
    ax.plot(tokens, predicted_labels, marker='o',
            linestyle='-', label='Predicted Label', color='blue', linewidth=0.5)
    ax.plot(tokens, true_labels, marker='o',
            linestyle='-', label='True Label', color='red', linewidth=1)


    ax.set_xlabel('Token Index')
    ax.set_ylabel('Label')
    ax.set_title('Sequence Labels')

    ax.legend()
    if save_path:
        plt.savefig(save_path, dpi=900)
    plt.show()


def plot_loss_curve(epoch_losses, save_path=None):
    plt.plot(epoch_losses)
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if save_path:
        plt.savefig(save_path, dpi=900)
    plt.show()


def plot_batch_losses(batch_losses, save_path=None):
    plt.plot(batch_losses)
    plt.title('Training Batch Loss Curve')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    if save_path:
        plt.savefig(save_path, dpi=900)
    plt.show()
