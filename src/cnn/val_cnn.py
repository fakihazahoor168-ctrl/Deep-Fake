# src/cnn/val_cnn.py
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cnn_model import CNNModel
from utils import device, plot_confusion_matrix, print_classification_report

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

val_dataset = datasets.ImageFolder('../../dataset/val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

num_classes = len(val_dataset.classes)
model = CNNModel(num_classes).to(device)
checkpoint = '../../models/cnn/best_cnn_model.pth'
state_dict = torch.load(checkpoint, map_location=device)
model.load_state_dict(state_dict)
model.eval()
print("Model loaded successfully.")

y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs,1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

plot_confusion_matrix(y_true, y_pred, val_dataset.classes, '../../comparison/comparison_results/cnn_val_confusion.png')
print_classification_report(y_true, y_pred, val_dataset.classes)
print("Validation complete.")
