import os
import sys
import django
import logging
import argparse

# Set up Django environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'product_verification.settings')
django.setup()

from products.model_training import train_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Train the product verification model"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train the product verification model')
    parser.add_argument('--data_dir', type=str, default='dataset', help='Directory containing the dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    
    args = parser.parse_args()
    
    # Train the model
    logger.info(f"Training model with data from {args.data_dir}...")
    model = train_model(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )
    
    if model:
        logger.info("Model training completed successfully!")
    else:
        logger.error("Model training failed!")

if __name__ == "__main__":
    main() 