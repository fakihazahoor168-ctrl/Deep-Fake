import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import json

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Save training history
def save_history(history, filename="history.json"):
    with open(filename, "w") as f:
        json.dump(history, f)

# Plot loss and accuracy curves
def plot_curves(history, title="Training Curves", save_path=None):
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title("Accuracy Curve")
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    plt.show()

# Confusion Matrix without seaborn
def plot_confusion_matrix(y_true, y_pred, classes, save_path=None):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Add numbers in matrix
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, cm[i][j], ha='center', va='center')

    plt.ylabel("Actual")
    plt.xlabel("Predicted")

    if save_path:
        plt.savefig(save_path)

    plt.show()

# Classification report
def print_classification_report(y_true, y_pred, classes):
    print(classification_report(y_true, y_pred, target_names=classes))
