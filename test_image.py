#!/usr/bin/env python
"""
Simple script to test the image analysis functionality.
Run this script with a path to an image file to test the analysis.
"""

import os
import sys
import django
import logging
from PIL import Image
import numpy as np

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'product_verification.settings')
django.setup()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the verification service
from products.verification_service import VerificationService

def test_image(image_path):
    """
    Test the image analysis functionality with a sample image
    
    Args:
        image_path (str): Path to the image file to analyze
    """
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return
        
        # Initialize verification service
        service = VerificationService()
        
        # Open the image
        with open(image_path, 'rb') as f:
            # Create a Django file-like object
            from django.core.files.base import ContentFile
            image_file = ContentFile(f.read())
            image_file.name = os.path.basename(image_path)
        
        # Analyze the image
        logger.info(f"Analyzing image: {image_path}")
        result = service.verify_image(image_file)
        
        # Print results
        print("\n=== Image Analysis Results ===")
        print(f"Status: {result['status']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Message: {result['message']}")
        
        # Print security features
        if 'security_features' in result:
            print("\nSecurity Features:")
            for feature, details in result['security_features'].items():
                print(f"- {feature}: {'Present' if details['present'] else 'Not present'} (Confidence: {details['confidence']:.2f})")
        
        # Print top predictions
        if 'top_predictions' in result:
            print("\nTop Predictions:")
            for pred in result['top_predictions']:
                print(f"- {pred['label']}: {pred['probability']:.2f}")
        
        # Print image info if available
        if 'image_info' in result:
            print("\nImage Information:")
            for key, value in result['image_info'].items():
                print(f"- {key}: {value}")
        
        print("\n=== End of Results ===")
        
    except Exception as e:
        logger.error(f"Error during image analysis: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_image.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    test_image(image_path) 