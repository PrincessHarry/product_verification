import os
import sys
import django
import logging
from datetime import datetime

# Set up Django environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'product_verification.settings')
django.setup()

from products.data_collection import DataCollector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main function to collect initial dataset"""
    try:
        # Initialize data collector
        collector = DataCollector()
        
        # Test with just a few products first
        test_products = [
            "Samsung Galaxy S21",
            "iPhone 13",
            "Nike Air Max"
        ]
        
        # Collect images for each product
        logger.info("Starting product image collection...")
        for product in test_products:
            try:
                logger.info(f"Processing {product}...")
                collector.collect_product_images(product, max_images=5)
            except Exception as e:
                logger.error(f"Error processing {product}: {str(e)}")
                continue
        
        logger.info("Initial data collection completed successfully")
        
    except Exception as e:
        logger.error(f"Error in data collection: {str(e)}")
        raise

if __name__ == "__main__":
    main()  