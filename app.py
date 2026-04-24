import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# --- ARSITEKTUR U-NET ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(128, 64)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(F.max_pool2d(x1, 2))
        x3 = self.down3(F.max_pool2d(x2, 2))
        x4 = self.down4(F.max_pool2d(x3, 2))
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.up_conv1(x)
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up_conv2(x)
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up_conv3(x)
        return self.out_conv(x)


@st.cache_resource
def load_model():
    model = UNet()
    
    model.load_state_dict(torch.load("unet_human_segmentation.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

st.set_page_config(page_title="U-Net Human Segmentation", page_icon="🤖", layout="centered")

st.title("Full-Body Human Movement Segmentation")
st.write("Upload an image of a person (preferably full-body) to see the U-Net model generate a segmentation mask.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert('RGB')
    
    
    img_resized = TF.resize(image, (128, 128))
    img_tensor = TF.to_tensor(img_resized)
    img_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_tensor_norm = img_norm(img_tensor).unsqueeze(0)
    
    with st.spinner('Generating mask...'):
        # Prediksi AI
        with torch.no_grad():
            pred = model(img_tensor_norm)
            pred_mask = (torch.sigmoid(pred[0][0]) > 0.5).float().numpy()

    st.success('Done!')
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(img_resized, use_container_width=True)
    with col2:
        st.subheader("AI Segmentation Mask")
        st.image(pred_mask, use_container_width=True, clamp=True)