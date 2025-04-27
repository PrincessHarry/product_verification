#!/usr/bin/env python
"""
Script to verify products using the trained model.
Usage: python verify_product.py path/to/image.jpg
"""

import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductVerificationModel(nn.Module):
    def __init__(self, num_classes=2):
        super(ProductVerificationModel, self).__init__()
        self.base_model = models.mobilenet_v2(pretrained=True)
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.base_model(x)

class ProductVerifier:
    def __init__(self, model_path="dataset/product_verification_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def _load_model(self, model_path):
        """Load the trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        # Initialize model
        model = ProductVerificationModel(num_classes=2)
        
        # Load trained weights
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def verify_image(self, image_path):
        """Verify a product image"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Get prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][prediction].item()

            # Get result
            result = "Original" if prediction == 1 else "Fake"
            confidence_percentage = confidence * 100

            return {
                "result": result,
                "confidence": confidence_percentage,
                "original_probability": probabilities[0][1].item() * 100,
                "fake_probability": probabilities[0][0].item() * 100
            }

        except Exception as e:
            logger.error(f"Error verifying image: {str(e)}")
            return None

def main():
    parser = argparse.ArgumentParser(description='Verify a product image')
    parser.add_argument('image_path', type=str, help='Path to the product image')
    parser.add_argument('--model', type=str, default='dataset/product_verification_model.pth',
                      help='Path to the trained model')
    
    args = parser.parse_args()
    
    try:
        verifier = ProductVerifier(args.model)
        result = verifier.verify_image(args.image_path)
        
        if result:
            print("\nVerification Results:")
            print(f"Product is: {result['result']}")
            print(f"Confidence: {result['confidence']:.2f}%")
            print(f"Original Probability: {result['original_probability']:.2f}%")
            print(f"Fake Probability: {result['fake_probability']:.2f}%")
        else:
            print("Failed to verify the image.")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 