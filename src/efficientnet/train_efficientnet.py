import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet
import sys, os

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import device, save_history, plot_curves, plot_confusion_matrix, print_classification_report

# ---------------- Device Info ----------------
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Training on CPU")

# ---------------- Data ----------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder('../../dataset/train', transform=transform)
val_dataset   = datasets.ImageFolder('../../dataset/val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False)

# ---------------- Model ----------------
model = EfficientNet.from_pretrained('efficientnet-b3')
model._fc = nn.Linear(model._fc.in_features, len(train_dataset.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 15  # updated epochs

history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
best_val_acc = 0.0

for epoch in range(num_epochs):

    # ----- TRAIN -----
    model.train()
    running_loss, running_corrects = 0.0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)

    train_loss = running_loss / len(train_dataset)
    train_acc  = running_corrects.double() / len(train_dataset)

    # ----- VALIDATION -----
    model.eval()
    val_loss_sum, val_corrects = 0.0, 0
    y_true, y_pred = [], []  # ✅ Correctly initialized

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss_sum += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_corrects += torch.sum(preds == labels.data)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    val_loss = val_loss_sum / len(val_dataset)
    val_acc  = val_corrects.double() / len(val_dataset)

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), '../../models/efficientnet/efficientnet_b3.pth')
        print("Best model saved.")

    # Save history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc.item())
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc.item())

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

# ---------------- SAVE RESULTS ----------------
save_history(history, '../../models/efficientnet/history.json')
plot_curves(history, save_path='../../models/efficientnet/curve.png')

plot_confusion_matrix(
    y_true, y_pred,
    classes=train_dataset.classes,
    save_path='../../comparison/comparison_results/efficientnet_val_confusion.png'
)

print_classification_report(y_true, y_pred, classes=train_dataset.classes)
