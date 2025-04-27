from django.core.management.base import BaseCommand
from products.train_model import main as train_model_main
import os
import sys
import torch
import torch.nn as nn
import torchvision.models as models

class Command(BaseCommand):
    help = 'Train the product verification model'

    def add_arguments(self, parser):
        parser.add_argument('--data_dir', type=str, default='dataset',
                          help='Directory containing the dataset')
        parser.add_argument('--batch_size', type=int, default=32,
                          help='Batch size for training')
        parser.add_argument('--epochs', type=int, default=10,
                          help='Number of epochs to train')
        parser.add_argument('--learning_rate', type=float, default=0.001,
                          help='Learning rate for training')

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Starting model training...'))
        
        try:
            # Set environment variables for the training script
            os.environ['DATA_DIR'] = options['data_dir']
            os.environ['BATCH_SIZE'] = str(options['batch_size'])
            os.environ['EPOCHS'] = str(options['epochs'])
            os.environ['LEARNING_RATE'] = str(options['learning_rate'])
            
            # Run the training script
            train_model_main()
            self.stdout.write(self.style.SUCCESS('Model training completed successfully!'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error during model training: {str(e)}'))
            sys.exit(1)

class ProductVerificationModel(nn.Module):
    def __init__(self, num_classes=2, model_type="mobilenet"):
        super(ProductVerificationModel, self).__init__()
        
        if model_type == "mobilenet":
            self.base_model = models.mobilenet_v2(pretrained=True)
            num_features = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(num_features, num_classes)
            )
        else:
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

# Create the model with the same architecture
model = ProductVerificationModel(num_classes=2, model_type="mobilenet")

# Load the state dictionary
model.load_state_dict(torch.load('dataset/product_verification_model.pth')) 