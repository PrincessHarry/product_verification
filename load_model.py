#!/usr/bin/env python
"""
Script to load the trained product verification model.
Run this script directly: python load_model.py
"""

import os
import torch
import torch.nn as nn
from torchvision import models
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductVerificationModel(nn.Module):
    def __init__(self, num_classes=2, model_type="mobilenet"):
        super(ProductVerificationModel, self).__init__()
        
        # Use a smaller, faster model
        if model_type == "mobilenet":
            # MobileNetV2 is much faster than ResNet50
            self.base_model = models.mobilenet_v2(pretrained=True)
            num_features = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(num_features, num_classes)
            )
        else:
            # Fallback to ResNet18 (smaller than ResNet50)
            self.base_model = models.resnet18(pretrained=True)
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
    
    def forward(self, x):
        return self.base_model(x)

def load_model(model_path="dataset/product_verification_model.pth", model_type="mobilenet"):
    """Load the trained model with the correct architecture"""
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return None
    
    # Create model with the same architecture used during training
    model = ProductVerificationModel(num_classes=2, model_type=model_type)
    
    # Load the state dictionary
    try:
        model.load_state_dict(torch.load(model_path))
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

if __name__ == "__main__":
    # Load the model
    model = load_model()
    
    if model:
        # Print model architecture
        logger.info("Model architecture:")
        logger.info(model)
        
        # Test the model with a random input
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        # Create a random input tensor
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        
        # Forward pass
        with torch.no_grad():
            output = model(dummy_input)
        
        logger.info(f"Model output shape: {output.shape}")
        logger.info(f"Model output: {output}")
        logger.info("Model loaded and tested successfully!")
    else:
        logger.error("Failed to load model.") 