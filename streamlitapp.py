import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import os

# Define the same CNN structure used for training
class FashionMNIST_CNN(nn.Module):
    def __init__(self):
        super(FashionMNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 64 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Streamlit interface
st.title("Fashion MNIST Classifier")

# Load the model with error handling
model_path = "fashion_mnist_cnn.pth"
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}. Please upload the correct file.")
else:
    model = FashionMNIST_CNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Preprocess the uploaded image
        image = Image.open(uploaded_file).convert("L")
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        image = transform(image).unsqueeze(0)

        # Make predictions
        output = model(image)
        predicted_label = output.argmax(dim=1).item()
        st.write(f"Predicted Label: {predicted_label}")