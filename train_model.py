#!/usr/bin/env python
"""
Standalone script to train the product verification model.
Run this script directly: python train_model.py
"""

import os
import sys
import django
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import json
import logging
from typing import Dict, List, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
import argparse

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'product_verification.settings')
django.setup()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductVerificationDataset(Dataset):
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

class ModelTrainer:
    def __init__(self, base_dir: str = "dataset", model_type: str = "mobilenet"):
        self.base_dir = base_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
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

    def prepare_data_loaders(self, image_paths: List[str], labels: List[int],
                           batch_size: int = 32, val_split: float = 0.2):
        """Prepare train and validation data loaders"""
        # Split the data
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=val_split, random_state=42, stratify=labels
        )

        # Create datasets
        train_dataset = ProductVerificationDataset(train_paths, train_labels, self.transform)
        val_dataset = ProductVerificationDataset(val_paths, val_labels, self.transform)

        # Create data loaders with reduced number of workers
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                num_workers=0)  # Set to 0 to avoid multiprocessing issues
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=0)  # Set to 0 to avoid multiprocessing issues

        return train_loader, val_loader

    def train(self, num_epochs: int = 5, batch_size: int = 16, learning_rate: float = 0.001):
        """Train the model with optimized parameters"""
        # Load dataset
        image_paths, labels = self.load_dataset()
        
        # Use a smaller validation split for faster training
        train_loader, val_loader = self.prepare_data_loaders(
            image_paths, labels, batch_size, val_split=0.1
        )

        # Initialize model
        model = ProductVerificationModel(num_classes=2, model_type=self.model_type).to(self.device)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop with early stopping
        best_val_acc = 0.0
        patience = 3
        patience_counter = 0
        
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
            logger.info(f'Train Loss: {train_loss/len(train_loader):.4f}, '
                       f'Train Acc: {train_acc:.2f}%')
            logger.info(f'Val Loss: {val_loss/len(val_loader):.4f}, '
                       f'Val Acc: {val_acc:.2f}%')

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 
                         os.path.join(self.base_dir, 'product_verification_model.pth'))
                logger.info(f'New best model saved with validation accuracy: {val_acc:.2f}%')
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= patience:
                logger.info(f'Early stopping after {epoch+1} epochs')
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train product verification model')
    parser.add_argument('--model', type=str, default='mobilenet', 
                        choices=['mobilenet', 'resnet18'],
                        help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    
    args = parser.parse_args()
    
    trainer = ModelTrainer(model_type=args.model)
    trainer.train(num_epochs=args.epochs, batch_size=args.batch_size, 
                 learning_rate=args.lr) 