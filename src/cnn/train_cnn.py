# src/cnn/train_cnn.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cnn_model import CNNModel
from utils import device, print_classification_report, plot_confusion_matrix, plot_curves

# ----------------- Device Info -----------------
print(f"\n🔥 Using Device for Training: {device}\n")

# ----------------- Transformations -----------------
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# ----------------- Dataset & Loader -----------------
train_dataset = datasets.ImageFolder('../../dataset/train', transform=train_transform)
val_dataset = datasets.ImageFolder('../../dataset/val', transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# ----------------- Model, Loss, Optimizer -----------------
num_classes = len(train_dataset.classes)
model = CNNModel(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

# ----------------- Training Setup -----------------
num_epochs = 25
best_val_acc = 0
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

print("\n🚀 Training Started...\n")

# ----------------- Training Loop -----------------
for epoch in range(num_epochs):
    # ----- Training -----
    model.train()
    running_loss = 0
    correct, total = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs,1)
        correct += (preds==labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

    # ----- Validation -----
    model.eval()
    val_correct, val_total = 0,0
    val_loss_sum = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss_sum += loss.item()
            _, preds = torch.max(outputs,1)
            val_correct += (preds==labels).sum().item()
            val_total += labels.size(0)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    val_loss = val_loss_sum / len(val_loader)
    val_acc = val_correct / val_total
    print(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}")

    # ----- Save best model -----
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), '../../models/cnn/best_cnn_model.pth')
        print("💾 Saved best model.")

    # ----- Scheduler step -----
    scheduler.step(val_loss)

    # ----- Update history -----
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

# ----------------- Save History -----------------
import json
with open('../../models/cnn/history.json', 'w') as f:
    json.dump(history, f)

# ----------------- Confusion Matrix & Report -----------------
plot_confusion_matrix(y_true, y_pred, val_dataset.classes, '../../comparison/comparison_results/cnn_val_confusion.png')
print_classification_report(y_true, y_pred, val_dataset.classes)

# ----------------- Plot Loss & Accuracy Curves -----------------
plot_curves(history, save_path='../../models/cnn/curve.png')

print("\n🎉 Training Complete.\n")
