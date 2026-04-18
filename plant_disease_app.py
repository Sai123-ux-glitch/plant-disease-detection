import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os


st.set_page_config(
    page_title="🌿 Plant Disease Detector",
    page_icon="🌱",
    layout="centered",
)


CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

NUM_CLASSES = len(CLASS_NAMES)   # 38


def ConvBlock(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1  = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2  = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = nn.Sequential(nn.MaxPool2d(4), nn.Flatten(), nn.Linear(512, num_diseases))

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        return self.classifier(out)



@st.cache_resource
def load_model(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ResNet9(3, NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device



def predict(image: Image.Image, model, device):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    tensor = transform(image).unsqueeze(0).to(device)   # (1, 3, 256, 256)
    with torch.no_grad():
        logits = model(tensor)                           # (1, 38)
        probs  = F.softmax(logits, dim=1)[0]             # (38,)
    top5_probs, top5_idx = torch.topk(probs, 5)
    return (
        CLASS_NAMES[top5_idx[0].item()],
        top5_probs[0].item() * 100,
        [(CLASS_NAMES[i.item()], p.item() * 100) for i, p in zip(top5_idx, top5_probs)],
    )


def format_label(raw: str) -> tuple[str, str]:
    """Returns (plant, condition) from 'Apple___Apple_scab' style label."""
    parts = raw.split("___")
    plant = parts[0].replace("_", " ")
    cond  = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
    return plant, cond



st.title("🌿 Plant Disease Detector")
st.markdown("Upload a leaf image to detect plant diseases using a **ResNet9** model trained on the *New Plant Diseases Dataset* (38 classes).")


st.sidebar.header("⚙️ Configuration")
default_path = "/content/best_model/best_model.pth"
model_path   = st.sidebar.text_input("Model path (.pth)", value=default_path)


if not os.path.exists(model_path):
    st.sidebar.error(f"Model not found at:\n`{model_path}`")
    st.info("👈 Please update the model path in the sidebar, or make sure `best_model/best_model.pth` exists in the working directory.")
    st.stop()

with st.spinner("Loading model …"):
    model, device = load_model(model_path)

st.sidebar.success(f"✅ Model loaded  |  Device: `{device}`")


uploaded = st.file_uploader("📂 Upload a leaf image", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        with st.spinner("Analysing …"):
            pred_class, confidence, top5 = predict(image, model, device)

        plant, condition = format_label(pred_class)

        if condition.lower() == "healthy":
            st.success(f"### ✅ {plant}")
            st.markdown(f"**Status:** Healthy  \n**Confidence:** `{confidence:.2f}%`")
        else:
            st.error(f"### ⚠️ {plant}")
            st.markdown(f"**Disease:** {condition}  \n**Confidence:** `{confidence:.2f}%`")

        st.markdown("---")
        st.markdown("#### 🔍 Top 5 Predictions")
        for name, prob in top5:
            p, c = format_label(name)
            label = f"{p} — {c}"
            st.progress(int(prob), text=f"{label}  `{prob:.1f}%`")


with st.expander("ℹ️ About this app"):
    st.markdown("""
**Model:** ResNet9 (custom residual CNN)  
**Dataset:** New Plant Diseases Dataset (Kaggle) — 87,000+ images, 38 classes  
**Plants covered:** Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato  
**Input size:** 256 × 256 RGB  
**Training:** Adam optimizer, OneCycleLR scheduler  

Upload a clear, well-lit photo of a single leaf for best results.
""")