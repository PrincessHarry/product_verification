#!/usr/bin/env python
"""
Script to train a model on Nigerian products for verification.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import logging
import argparse
from typing import List, Tuple
import json
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NigerianProductDataset(Dataset):
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

class NigerianProductModel(nn.Module):
    def __init__(self, num_classes=2):
        super(NigerianProductModel, self).__init__()
        
        # Use MobileNetV2 as it's faster and lighter
        self.base_model = models.mobilenet_v2(pretrained=True)
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.base_model(x)

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
    
    def train(self, num_epochs=10, batch_size=8, learning_rate=0.001):
        """Train the model"""
        # Load dataset
        image_paths, labels = self.load_dataset()
        
        if len(image_paths) == 0:
            logger.error("No images found in the dataset. Please collect data first.")
            return None
        
        logger.info(f"Found {len(image_paths)} images in the dataset")
        
        # Create dataset
        dataset = NigerianProductDataset(image_paths, labels, self.transform)
        
        # Split dataset into train and validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        model = NigerianProductModel(num_classes=2).to(self.device)
        
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
                torch.save(model.state_dict(), os.path.join(self.base_dir, 'nigerian_product_model.pth'))
                logger.info(f'New best model saved with validation accuracy: {val_acc:.2f}%')
        
        return model

def main():
    parser = argparse.ArgumentParser(description='Train Nigerian product verification model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    trainer = ModelTrainer()
    trainer.train(num_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)

if __name__ == "__main__":
    main() 