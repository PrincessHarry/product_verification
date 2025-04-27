#!/usr/bin/env python
"""
Script to verify Nigerian products using the trained model.
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import logging
import argparse
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NigerianProductModel(nn.Module):
    def __init__(self, num_classes=2):
        super(NigerianProductModel, self).__init__()
        # Use MobileNetV2 as base model
        self.base_model = models.mobilenet_v2(pretrained=False)
        
        # Modify the classifier
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.base_model(x)

class ProductVerifier:
    def __init__(self, model_path, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Initialize model
        self.model = NigerianProductModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def verify_image(self, image_path):
        """
        Verify a product image and return the prediction and confidence.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            tuple: (prediction, confidence)
                - prediction (str): 'original' or 'fake'
                - confidence (float): Confidence score between 0 and 1
        """
        try:
            # Load and transform image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                prediction = 'original' if predicted.item() == 1 else 'fake'
                confidence = confidence.item()
                
                return prediction, confidence
                
        except Exception as e:
            logger.error(f"Error verifying image {image_path}: {str(e)}")
            return None, None

def main():
    parser = argparse.ArgumentParser(description='Verify Nigerian products')
    parser.add_argument('--model_path', type=str, default='nigerian_product_model.pth',
                      help='Path to the trained model')
    parser.add_argument('--image_path', type=str, required=True,
                      help='Path to the image to verify')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found: {args.model_path}")
        return
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        logger.error(f"Image file not found: {args.image_path}")
        return
    
    # Initialize verifier
    verifier = ProductVerifier(args.model_path)
    
    # Verify image
    prediction, confidence = verifier.verify_image(args.image_path)
    
    if prediction is not None:
        logger.info(f"Product Verification Result:")
        logger.info(f"Prediction: {prediction}")
        logger.info(f"Confidence: {confidence:.2%}")
    else:
        logger.error("Verification failed")

if __name__ == "__main__":
    main() 