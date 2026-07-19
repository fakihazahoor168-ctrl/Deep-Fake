# src/cnn/train_cnn.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cnn_model import CNNModel
from utils import device, print_classification_report, plot_confusion_matrix

# ----------------- Device Message -----------------
print(f"\n🔥 Using Device for Training: {device}\n")

# ----------------- Transform -----------------
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

# ----------------- Model -----------------
num_classes = len(train_dataset.classes)
model = CNNModel(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

print("\n🚀 Training Started...\n")

# ----------------- Training Loop -----------------
num_epochs = 25
best_val_acc = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    correct, total = 0,0

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

    train_acc = correct/total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}")

    # Validation
    model.eval()
    val_correct, val_total = 0,0
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs,1)
            val_correct += (preds==labels).sum().item()
            val_total += labels.size(0)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    val_acc = val_correct/val_total
    print(f"Validation Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), '../../models/cnn/best_cnn_model.pth')
        print("💾 Saved best model.")

    scheduler.step(1 - val_acc)

# Confusion Matrix and Report
plot_confusion_matrix(y_true, y_pred, val_dataset.classes, '../../comparison/comparison_results/cnn_val_confusion.png')
print_classification_report(y_true, y_pred, val_dataset.classes)
print("\n🎉 Training complete.\n")
