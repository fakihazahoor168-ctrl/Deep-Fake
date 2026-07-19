# comparison/compare_models.py

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os, sys
import matplotlib.pyplot as plt

# ---------------- Path Fix ----------------
# Add src folder to path so utils.py can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from utils import device, plot_confusion_matrix, print_classification_report, plot_curves
from cnn.cnn_model import CNNModel
from efficientnet_pytorch import EfficientNet

# ----------------- Device Info -----------------
print(f"🔥 Using device: {device}\n")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
else:
    print("Training on CPU\n")

# ----------------- Dataset -----------------
val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

val_dataset = datasets.ImageFolder('../dataset/val', transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# ----------------- Load Models -----------------
num_classes = len(val_dataset.classes)

# CNN
cnn_model = CNNModel(num_classes).to(device)
cnn_model.load_state_dict(torch.load('../models/cnn/best_cnn_model.pth'))
cnn_model.eval()

# EfficientNet-B3
efficient_model = EfficientNet.from_pretrained('efficientnet-b3')
efficient_model._fc = torch.nn.Linear(efficient_model._fc.in_features, num_classes)
efficient_model.load_state_dict(torch.load('../models/efficientnet/efficientnet_b3.pth'))
efficient_model = efficient_model.to(device)
efficient_model.eval()

# ----------------- Evaluation Function -----------------
def evaluate_model(model, loader, criterion=None):
    y_true, y_pred = [], []
    total_loss, total = 0.0, 0
    correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if criterion:
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = correct / total
    avg_loss = total_loss / total if criterion else 0
    return y_true, y_pred, acc, avg_loss

# ----------------- Evaluate CNN -----------------
criterion = torch.nn.CrossEntropyLoss()
y_true_cnn, y_pred_cnn, cnn_acc, cnn_loss = evaluate_model(cnn_model, val_loader, criterion)
print(f"\nCNN Accuracy: {cnn_acc:.4f}, Loss: {cnn_loss:.4f}")
plot_confusion_matrix(y_true_cnn, y_pred_cnn, val_dataset.classes, '../comparison/comparison_results/cnn_val_confusion.png')
print_classification_report(y_true_cnn, y_pred_cnn, val_dataset.classes)

# ----------------- Evaluate EfficientNet -----------------
y_true_eff, y_pred_eff, eff_acc, eff_loss = evaluate_model(efficient_model, val_loader, criterion)
print(f"\nEfficientNet-B3 Accuracy: {eff_acc:.4f}, Loss: {eff_loss:.4f}")
plot_confusion_matrix(y_true_eff, y_pred_eff, val_dataset.classes, '../comparison/comparison_results/efficientnet_val_confusion.png')
print_classification_report(y_true_eff, y_pred_eff, val_dataset.classes)

# ----------------- Accuracy Comparison Plot -----------------
plt.figure(figsize=(6,4))
plt.bar(['CNN', 'EfficientNet-B3'], [cnn_acc, eff_acc], color=['skyblue','salmon'])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.savefig('../comparison/comparison_results/accuracy_comparison.png')
plt.show()
