# app.py
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Define same Generator
class Generator(nn.Module):
    def __init__(self, z_dim=100, img_dim=28*28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + 10, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, img_dim),
            nn.Tanh()
        )

    def forward(self, z, labels):
        one_hot = torch.nn.functional.one_hot(labels, 10).float()
        input = torch.cat([z, one_hot], dim=1)
        return self.net(input)

# Load model
z_dim = 100
device = 'cpu'
generator = Generator(z_dim)
generator.load_state_dict(torch.load("generator.pth", map_location=device))
generator.eval()

st.title("Handwritten Digit Generator (0–9)")

digit = st.selectbox("Select digit to generate", list(range(10)))
generate_button = st.button("Generate")

if generate_button:
    noise = torch.randn(5, z_dim)
    labels = torch.tensor([digit] * 5)
    with torch.no_grad():
        generated = generator(noise, labels)
    images = generated.view(-1, 28, 28).detach().numpy()

    # Plot
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axes[i].imshow((images[i] + 1) / 2, cmap='gray')
        axes[i].axis('off')
    st.pyplot(fig)
