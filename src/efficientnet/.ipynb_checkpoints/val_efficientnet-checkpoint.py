# src/efficientnet/val_efficientnet.py

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet
import sys, os

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import device, plot_confusion_matrix, print_classification_report

# ---------------- Data ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

val_dataset = datasets.ImageFolder('../../dataset/val', transform=transform)
val_loader  = DataLoader(val_dataset, batch_size=16, shuffle=False)

# ---------------- Model ----------------
model = EfficientNet.from_pretrained('efficientnet-b3')
model._fc = torch.nn.Linear(model._fc.in_features, len(val_dataset.classes))
model.load_state_dict(torch.load('../../models/efficientnet/efficientnet_b3.pth'))
model = model.to(device)
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Save confusion matrix
plot_confusion_matrix(
    y_true, y_pred,
    classes=val_dataset.classes,
    save_path='../../comparison/comparison_results/efficientnet_val_confusion.png'
)

print_classification_report(y_true, y_pred, val_dataset.classes)
