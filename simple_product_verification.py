#!/usr/bin/env python
"""
Simple Product Verification System
This script handles data collection, model training, and product verification in one file.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import requests
from io import BytesIO
import logging
import argparse
import time
import random
from typing import List, Dict, Tuple
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the model architecture
class ProductVerificationModel(nn.Module):
    def __init__(self, num_classes=2):
        super(ProductVerificationModel, self).__init__()
        
        # Use MobileNetV2 as it's faster and lighter
        self.base_model = models.mobilenet_v2(pretrained=True)
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.base_model(x)

# Define the dataset class
class ProductDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            # Return a default image and label in case of error
            return torch.zeros((3, 224, 224)), label

# Define the data collector class
class DataCollector:
    def __init__(self, base_dir="dataset"):
        self.base_dir = base_dir
        self.images_dir = os.path.join(base_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Define common products with their image URLs
        self.products = {
            "iPhone": {
                "original": [
                    "https://store.storeimages.cdn-apple.com/4982/as-images.apple.com/is/iphone-13-blue-select-2021?wid=940&hei=1112&fmt=png-alpha&.v=1645572386470",
                    "https://store.storeimages.cdn-apple.com/4982/as-images.apple.com/is/iphone-13-mini-blue-select-2021?wid=940&hei=1112&fmt=png-alpha&.v=1645572386470",
                    "https://store.storeimages.cdn-apple.com/4982/as-images.apple.com/is/iphone-13-pro-max-graphite-select?wid=940&hei=1112&fmt=png-alpha&.v=1645552346280"
                ],
                "fake": [
                    "https://i.ebayimg.com/images/g/0~AAAOSw~1Fj~y~Y/s-l1600.jpg",
                    "https://i.ebayimg.com/images/g/0~AAAOSw~1Fj~y~Y/s-l1600.jpg",
                    "https://i.ebayimg.com/images/g/0~AAAOSw~1Fj~y~Y/s-l1600.jpg"
                ]
            },
            "Nike_Air_Max": {
                "original": [
                    "https://static.nike.com/a/images/t_PDP_1280_v1/f_auto,q_auto:eco/skwgyqrbfzhu6uyeh0gg/air-max-270-shoes-V4DfZQ.png",
                    "https://static.nike.com/a/images/t_PDP_1280_v1/f_auto,q_auto:eco/skwgyqrbfzhu6uyeh0gg/air-max-270-shoes-V4DfZQ.png",
                    "https://static.nike.com/a/images/t_PDP_1280_v1/f_auto,q_auto:eco/skwgyqrbfzhu6uyeh0gg/air-max-270-shoes-V4DfZQ.png"
                ],
                "fake": [
                    "https://i.ebayimg.com/images/g/0~AAAOSw~1Fj~y~Y/s-l1600.jpg",
                    "https://i.ebayimg.com/images/g/0~AAAOSw~1Fj~y~Y/s-l1600.jpg",
                    "https://i.ebayimg.com/images/g/0~AAAOSw~1Fj~y~Y/s-l1600.jpg"
                ]
            },
            "Fair_White_Cream": {
                "original": [
                    "https://ng.jumia.is/unsafe/fit-in/500x500/filters:fill(white)/product/82/482364/1.jpg",
                    "https://ng.jumia.is/unsafe/fit-in/500x500/filters:fill(white)/product/82/482364/2.jpg",
                    "https://ng.jumia.is/unsafe/fit-in/500x500/filters:fill(white)/product/82/482364/3.jpg"
                ],
                "fake": [
                    "https://i.ebayimg.com/images/g/MV8AAOSwPLNjwL6Y/s-l1600.jpg",
                    "https://i.ebayimg.com/images/g/dVwAAOSwOFVjwL6Y/s-l1600.jpg",
                    "https://i.ebayimg.com/images/g/2Z4AAOSwPLNjwL6Y/s-l1600.jpg"
                ]
            }
        }
    
    def download_image(self, url: str, save_path: str) -> bool:
        """Download an image from URL and save it"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            img = Image.open(BytesIO(response.content))
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            img.save(save_path, 'JPEG', quality=95)
            logger.info(f"Successfully downloaded: {save_path}")
            return True
        except Exception as e:
            logger.error(f"Error downloading {url}: {str(e)}")
            return False
    
    def collect_data(self):
        """Collect data for all products"""
        for product_name, images in self.products.items():
            logger.info(f"Collecting data for {product_name}")
            
            # Create directories
            product_dir = os.path.join(self.images_dir, product_name)
            original_dir = os.path.join(product_dir, "original")
            fake_dir = os.path.join(product_dir, "fake")
            
            os.makedirs(original_dir, exist_ok=True)
            os.makedirs(fake_dir, exist_ok=True)
            
            # Download original images
            for i, url in enumerate(images["original"]):
                save_path = os.path.join(original_dir, f"original_{i+1}.jpg")
                if not os.path.exists(save_path):
                    self.download_image(url, save_path)
                    time.sleep(1)  # Be nice to servers
            
            # Download fake images
            for i, url in enumerate(images["fake"]):
                save_path = os.path.join(fake_dir, f"fake_{i+1}.jpg")
                if not os.path.exists(save_path):
                    self.download_image(url, save_path)
                    time.sleep(1)  # Be nice to servers

# Define the model trainer class
class ModelTrainer:
    def __init__(self, base_dir="dataset"):
        self.base_dir = base_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_dataset(self) -> Tuple[List[str], List[int]]:
        """Load all images and their labels from the dataset"""
        image_paths = []
        labels = []

        # Walk through the dataset directory
        for root, dirs, files in os.walk(os.path.join(self.base_dir, "images")):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(root, file)
                    # Label is 1 for original, 0 for fake
                    label = 1 if 'original' in image_path else 0
                    image_paths.append(image_path)
                    labels.append(label)

        return image_paths, labels
    
    def train(self, num_epochs=5, batch_size=8, learning_rate=0.001):
        """Train the model"""
        # Load dataset
        image_paths, labels = self.load_dataset()
        
        if len(image_paths) == 0:
            logger.error("No images found in the dataset. Please collect data first.")
            return None
        
        logger.info(f"Found {len(image_paths)} images in the dataset")
        
        # Create dataset
        dataset = ProductDataset(image_paths, labels, self.transform)
        
        # Split dataset into train and validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        model = ProductVerificationModel(num_classes=2).to(self.device)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_acc = 100 * train_correct / train_total
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_acc = 100 * val_correct / val_total
            
            # Log progress
            logger.info(f'Epoch [{epoch+1}/{num_epochs}]')
            logger.info(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
            logger.info(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), os.path.join(self.base_dir, 'product_verification_model.pth'))
                logger.info(f'New best model saved with validation accuracy: {val_acc:.2f}%')
        
        return model

# Define the product verifier class
class ProductVerifier:
    def __init__(self, model_path="dataset/product_verification_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = ProductVerificationModel(num_classes=2).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def verify_image(self, image_path):
        """Verify if an image is of an original or fake product"""
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return None
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                # Get confidence scores
                confidence = probabilities[0][predicted.item()].item() * 100
                
                # Determine result
                result = "ORIGINAL" if predicted.item() == 1 else "FAKE"
                
                return {
                    "result": result,
                    "confidence": confidence,
                    "original_prob": probabilities[0][1].item() * 100,
                    "fake_prob": probabilities[0][0].item() * 100
                }
                
        except Exception as e:
            logger.error(f"Error verifying image: {str(e)}")
            return None

def main():
    parser = argparse.ArgumentParser(description='Simple Product Verification System')
    parser.add_argument('--mode', type=str, default='all', 
                        choices=['collect', 'train', 'verify', 'all'],
                        help='Mode to run the system in')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to image for verification')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    
    args = parser.parse_args()
    
    # Collect data
    if args.mode in ['collect', 'all']:
        logger.info("Collecting data...")
        collector = DataCollector()
        collector.collect_data()
    
    # Train model
    if args.mode in ['train', 'all']:
        logger.info("Training model...")
        trainer = ModelTrainer()
        trainer.train(num_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
    
    # Verify image
    if args.mode in ['verify', 'all'] and args.image:
        logger.info(f"Verifying image: {args.image}")
        verifier = ProductVerifier()
        result = verifier.verify_image(args.image)
        
        if result:
            print("\n" + "="*50)
            print(f"VERIFICATION RESULT: {result['result']}")
            print(f"Confidence: {result['confidence']:.2f}%")
            print(f"Original Probability: {result['original_prob']:.2f}%")
            print(f"Fake Probability: {result['fake_prob']:.2f}%")
            print("="*50 + "\n")
        else:
            print("Verification failed. Please check the image and try again.")

if __name__ == "__main__":
    main() 