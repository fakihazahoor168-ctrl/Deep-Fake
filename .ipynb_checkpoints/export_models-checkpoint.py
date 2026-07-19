# export_models.py
import torch
import torch.nn as nn
import os
import sys

# Fix import path for utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from utils import device
except ImportError:
    # Fallback if utils.device not available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

# ---------------- Model Definitions ----------------
# EfficientNet-B3
from efficientnet_pytorch import EfficientNet

class EfficientNetB3(nn.Module):
    def __init__(self, num_classes=2):
        super(EfficientNetB3, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b3')
        self.model._fc = nn.Linear(self.model._fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Simple CNN Example (adjust according to your trained CNN)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 256),  # Assuming input 224x224
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ---------------- Paths ----------------
model_paths = {
    'efficientnet_b3': os.path.join('models', 'efficientnet', 'efficientnet_b3.pth'),
    'cnn': os.path.join('models', 'cnn', 'cnn_model.pth')
}
export_dir = os.path.join('models', 'exported')
os.makedirs(export_dir, exist_ok=True)

# ---------------- Export Function ----------------
def export_model(model, model_name, input_size=(1,3,224,224)):
    model.to(device)
    model.eval()

    # TorchScript
    script_model_path = os.path.join(export_dir, f'{model_name}_script.pt')
    scripted_model = torch.jit.script(model)
    scripted_model.save(script_model_path)
    print(f"[{model_name}] TorchScript saved: {script_model_path}")

    # ONNX
    onnx_model_path = os.path.join(export_dir, f'{model_name}.onnx')
    dummy_input = torch.randn(*input_size, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"[{model_name}] ONNX saved: {onnx_model_path}")

# ---------------- Load & Export Models ----------------
# 1️⃣ EfficientNet-B3
num_classes_eff = 2  # change as per your dataset
eff_model = EfficientNetB3(num_classes=num_classes_eff)
eff_model.load_state_dict(torch.load(model_paths['efficientnet_b3'], map_location=device))
export_model(eff_model, 'efficientnet_b3')

# 2️⃣ CNN
num_classes_cnn = 2  # change as per your dataset
cnn_model = SimpleCNN(num_classes=num_classes_cnn)
cnn_model.load_state_dict(torch.load(model_paths['cnn'], map_location=device))
export_model(cnn_model, 'cnn')

print("All models exported successfully!")
