import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
import json
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductVerificationDataset(Dataset):
    def __init__(self, metadata_file, transform=None):
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
            
        self.transform = transform
        
        # Prepare data
        self.data = []
        self.labels = []
        
        # Add original products
        for product in self.metadata["products"]:
            self.data.append(product["original_image"])
            self.labels.append(1)  # 1 for original
            
            # Add fake products
            for fake_image in product["fake_images"]:
                self.data.append(fake_image)
                self.labels.append(0)  # 0 for fake
        
        # Add frames from videos
        for video in self.metadata["videos"]:
            for frame in video["frames"]:
                self.data.append(frame)
                # We don't know the label for video frames, so we'll use a placeholder
                self.labels.append(-1)  # -1 for unknown
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {str(e)}")
            # Return a blank image and label if there's an error
            return torch.zeros((3, 224, 224)), label

class ProductVerificationModel(nn.Module):
    def __init__(self, num_classes=2):
        super(ProductVerificationModel, self).__init__()
        
        # Use a pre-trained model as the base
        self.base_model = models.resnet50(pretrained=True)
        
        # Modify the final layer for our task
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.base_model(x)

def train_model(data_dir="dataset", batch_size=32, epochs=10, learning_rate=0.001):
    """Train the product verification model"""
    logger.info("Starting model training...")
    
    # Load metadata
    metadata_file = os.path.join(data_dir, "metadata.json")
    
    if not os.path.exists(metadata_file):
        logger.error(f"Metadata file not found: {metadata_file}")
        return None
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = ProductVerificationDataset(metadata_file, transform=transform)
    
    if len(dataset) == 0:
        logger.error("No data found in the dataset")
        return None
    
    # Split dataset
    train_data, val_data, train_labels, val_labels = train_test_split(
        dataset.data, dataset.labels, test_size=0.2, random_state=42
    )
    
    # Create data loaders
    train_dataset = ProductVerificationDataset(metadata_file, transform=transform)
    train_dataset.data = train_data
    train_dataset.labels = train_labels
    
    val_dataset = ProductVerificationDataset(metadata_file, transform=transform)
    val_dataset.data = val_data
    val_dataset.labels = val_labels
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = ProductVerificationModel().to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            # Skip unknown labels
            if -1 in labels:
                continue
                
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                # Skip unknown labels
                if -1 in labels:
                    continue
                    
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val Accuracy: {100 * correct / total:.2f}%")
    
    # Save the model
    model_path = os.path.join(data_dir, "product_verification_model.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    return model

def extract_features(image_path, model, transform):
    """Extract features from an image using the trained model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        
        # Get features from the model
        model.eval()
        with torch.no_grad():
            # Get features from the second-to-last layer
            features = model.base_model(image)
            features = features.cpu().numpy()
        
        return features
    except Exception as e:
        logger.error(f"Error extracting features from {image_path}: {str(e)}")
        return None 