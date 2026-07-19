import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import sys, os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("../src/cnn"))
from cnn_model import CNNModel

from efficientnet_pytorch import EfficientNet  # IMPORTANT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: **{device}**")

def load_efficientnet():
    model = EfficientNet.from_pretrained('efficientnet-b3')
    model._fc = nn.Linear(model._fc.in_features, 2)

    state = torch.load(
        "../models/efficientnet/efficientnet_b3.pth",
        map_location=device
    )
    model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model

def load_cnn():
    model = CNNModel(num_classes=2)

    state = torch.load(
        "../models/cnn/best_cnn_model.pth",
        map_location=device
    )
    model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def preprocess_image(image_file):
    image = Image.open(image_file).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor

def predict(model, tensor):
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_class = probs.argmax()
    return pred_class, probs  # return class index + probabilities

st.title("Deepfake Detection — CNN vs EfficientNet")
st.write("Upload an image to compare predictions from both models.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

@st.cache_resource
def load_models_once():
    return load_cnn(), load_efficientnet()

cnn_model, efficient_model = load_models_once()

class_names = ["FAKE", "REAL"]

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", width=300)

    tensor = preprocess_image(uploaded_file)

    cnn_pred, cnn_probs = predict(cnn_model, tensor)
    eff_pred, eff_probs = predict(efficient_model, tensor)

    st.subheader("🔍 Predictions")
    st.write(f"**CNN Model:** {class_names[cnn_pred]} ({cnn_probs[cnn_pred]*100:.2f}% confident)")
    st.write(f"**EfficientNet-B3 Model:** {class_names[eff_pred]} ({eff_probs[eff_pred]*100:.2f}% confident)")

    # ---------------- Plot Bar Graph ----------------
    fig, ax = plt.subplots()
    x = class_names
    ax.bar([i - 0.15 for i in range(2)], cnn_probs*100, width=0.3, label="CNN")
    ax.bar([i + 0.15 for i in range(2)], eff_probs*100, width=0.3, label="EfficientNet")
    ax.set_xticks(range(2))
    ax.set_xticklabels(class_names)
    ax.set_ylabel("Confidence (%)")
    ax.set_title("Prediction Confidence Comparison")
    ax.legend()
    st.pyplot(fig)

    st.success("✔ Predictions generated successfully!")

st.write("---")
st.write("Made with ❤️ | Deepfake Project")





