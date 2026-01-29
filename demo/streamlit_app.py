"""Streamlit demo for semi-supervised image classification."""

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import os
from typing import Dict, List, Tuple

from src.models.resnet import ResNet50
from src.utils.device import get_device
from src.data.cifar10 import CIFAR10DataModule


# Page configuration
st.set_page_config(
    page_title="Semi-supervised Image Classification",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CIFAR-10 class names
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Color mapping for classes
CLASS_COLORS = plt.cm.Set3(np.linspace(0, 1, len(CLASS_NAMES)))


@st.cache_resource
def load_model(checkpoint_path: str) -> ResNet50:
    """Load the trained model.
    
    Args:
        checkpoint_path: Path to model checkpoint.
        
    Returns:
        Loaded model.
    """
    device = get_device("auto")
    
    model = ResNet50(
        num_classes=len(CLASS_NAMES),
        pretrained=False,
        dropout=0.0,
        use_ema=False
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess image for model input.
    
    Args:
        image: PIL Image.
        
    Returns:
        Preprocessed tensor.
    """
    # Resize to 224x224
    image = image.resize((224, 224))
    
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Convert to tensor and normalize
    image_array = np.array(image) / 255.0
    image_tensor = torch.tensor(image_array, dtype=torch.float32).permute(2, 0, 1)
    
    # Normalize with ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    
    return image_tensor.unsqueeze(0)


def predict_image(model: ResNet50, image_tensor: torch.Tensor) -> Tuple[np.ndarray, str, float]:
    """Predict class for an image.
    
    Args:
        model: Trained model.
        image_tensor: Preprocessed image tensor.
        
    Returns:
        Tuple of (probabilities, predicted_class, confidence).
    """
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = F.softmax(logits, dim=1)
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class_idx].item()
    
    return probabilities.cpu().numpy()[0], CLASS_NAMES[predicted_class_idx], confidence


def plot_predictions(probabilities: np.ndarray, predicted_class: str, confidence: float) -> None:
    """Plot prediction results.
    
    Args:
        probabilities: Class probabilities.
        predicted_class: Predicted class name.
        confidence: Prediction confidence.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot of probabilities
    bars = ax1.bar(CLASS_NAMES, probabilities, color=CLASS_COLORS, alpha=0.7)
    ax1.set_xlabel("Class")
    ax1.set_ylabel("Probability")
    ax1.set_title("Class Probabilities")
    ax1.tick_params(axis='x', rotation=45)
    
    # Highlight predicted class
    predicted_idx = CLASS_NAMES.index(predicted_class)
    bars[predicted_idx].set_color('red')
    bars[predicted_idx].set_alpha(1.0)
    
    # Add confidence text
    ax1.text(0.02, 0.98, f"Predicted: {predicted_class}\nConfidence: {confidence:.3f}",
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Pie chart of top 5 predictions
    top5_indices = np.argsort(probabilities)[-5:]
    top5_probs = probabilities[top5_indices]
    top5_names = [CLASS_NAMES[i] for i in top5_indices]
    
    ax2.pie(top5_probs, labels=top5_names, autopct='%1.1f%%', startangle=90)
    ax2.set_title("Top 5 Predictions")
    
    plt.tight_layout()
    st.pyplot(fig)


def main():
    """Main Streamlit application."""
    st.title("🔬 Semi-supervised Image Classification")
    st.markdown("""
    This demo showcases a semi-supervised learning approach for image classification using CIFAR-10 dataset.
    The model uses pseudo-labeling and consistency regularization to learn from both labeled and unlabeled data.
    """)
    
    # Sidebar
    st.sidebar.header("Model Configuration")
    
    # Model selection
    checkpoint_path = st.sidebar.selectbox(
        "Select Model Checkpoint",
        options=["checkpoints/best_model.pth", "checkpoints/checkpoint_epoch_50.pth"],
        help="Choose a trained model checkpoint"
    )
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        st.error(f"Checkpoint not found: {checkpoint_path}")
        st.info("Please train a model first using the training script.")
        return
    
    # Load model
    try:
        model = load_model(checkpoint_path)
        st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📸 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image to classify"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Preprocess and predict
            image_tensor = preprocess_image(image)
            probabilities, predicted_class, confidence = predict_image(model, image_tensor)
            
            # Display results
            st.success(f"Predicted Class: **{predicted_class}**")
            st.info(f"Confidence: **{confidence:.3f}**")
            
            # Plot predictions
            plot_predictions(probabilities, predicted_class, confidence)
    
    with col2:
        st.header("📊 Model Information")
        
        # Model stats
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        st.metric("Total Parameters", f"{total_params:,}")
        st.metric("Trainable Parameters", f"{trainable_params:,}")
        
        # Device info
        device = next(model.parameters()).device
        st.metric("Device", str(device))
        
        # Class information
        st.subheader("CIFAR-10 Classes")
        for i, class_name in enumerate(CLASS_NAMES):
            st.write(f"{i}: {class_name}")
        
        # Model architecture
        st.subheader("Model Architecture")
        st.code("""
        ResNet-50 Backbone
        ├── Convolutional Layers
        ├── Batch Normalization
        ├── ReLU Activation
        ├── Max Pooling
        ├── Residual Blocks (x4)
        ├── Global Average Pooling
        └── Classification Head
            ├── Dropout
            └── Linear Layer (10 classes)
        """)
    
    # Additional information
    st.header("🔍 About Semi-supervised Learning")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Pseudo-labeling")
        st.markdown("""
        - Use model predictions on unlabeled data as pseudo-labels
        - Only use high-confidence predictions
        - Gradually increase pseudo-label threshold during training
        """)
    
    with col2:
        st.subheader("Consistency Regularization")
        st.markdown("""
        - Apply different augmentations to same image
        - Enforce consistent predictions
        - Use exponential moving average (EMA) of model weights
        """)
    
    with col3:
        st.subheader("Benefits")
        st.markdown("""
        - Learn from large amounts of unlabeled data
        - Reduce annotation costs
        - Improve model generalization
        - Better performance with limited labeled data
        """)


if __name__ == "__main__":
    main()
