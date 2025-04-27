import os
import shutil
import json
from typing import Dict, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetReorganizer:
    def __init__(self):
        self.base_dir = "dataset/images"
        self.metadata_path = "dataset/metadata.json"
        self.categories = {
            "Electronics": {
                "Smartphones": ["iPhone", "Samsung", "Xiaomi"],
                "Laptops": ["Apple", "Dell", "HP"],
                "Watches": ["Apple", "Samsung", "Rolex"]
            },
            "Fashion": {
                "Shoes": ["Nike", "Adidas", "Under Armour"],
                "Bags": ["Louis Vuitton", "Gucci", "Prada"],
                "Accessories": ["Rolex", "Cartier", "Hermes"]
            },
            "Cosmetics": {
                "Skincare": ["La Mer", "SK-II", "Estee Lauder", "Clinique"],
                "Makeup": ["MAC", "Dior", "Chanel", "NARS"],
                "Haircare": ["Kerastase", "Olaplex", "Living Proof"],
                "Bodycare": ["L'Occitane", "Jo Malone", "Molton Brown"],
                "Soap": ["Dove", "L'Occitane", "Aesop", "Fresh"],
                "Cream": ["La Mer", "La Prairie", "Tatcha", "Drunk Elephant"]
            }
        }

    def create_directory_structure(self):
        """Create the organized directory structure"""
        for category, subcategories in self.categories.items():
            for subcategory, brands in subcategories.items():
                for brand in brands:
                    # Create paths for both original and fake products
                    for product_type in ['original', 'fake']:
                        path = os.path.join(self.base_dir, category, subcategory, brand, product_type)
                        os.makedirs(path, exist_ok=True)
                        logger.info(f"Created directory: {path}")

    def move_existing_files(self):
        """Move existing files to their proper locations"""
        # Map of old directories to new locations
        directory_mapping = {
            "Samsung": "Electronics/Smartphones/Samsung",
            "Samsung Galaxy S21_880164000000": "Electronics/Smartphones/Samsung/Galaxy_S21",
            "Nike Air Max_194956000000": "Fashion/Shoes/Nike/Air_Max",
            "iPhone 13_190199000000": "Electronics/Smartphones/iPhone/iPhone_13"
        }

        for old_name, new_path in directory_mapping.items():
            old_path = os.path.join(self.base_dir, old_name)
            if os.path.exists(old_path):
                new_full_path = os.path.join(self.base_dir, new_path)
                os.makedirs(new_full_path, exist_ok=True)
                
                # Move all files from old directory to new location
                for root, dirs, files in os.walk(old_path):
                    for file in files:
                        old_file = os.path.join(root, file)
                        # Determine if it's an original or fake image
                        is_original = "original" in root.lower()
                        new_subdir = "original" if is_original else "fake"
                        new_file = os.path.join(new_full_path, new_subdir, file)
                        
                        # Create subdirectory if it doesn't exist
                        os.makedirs(os.path.dirname(new_file), exist_ok=True)
                        
                        # Move the file
                        shutil.move(old_file, new_file)
                        logger.info(f"Moved {old_file} to {new_file}")

    def cleanup_empty_directories(self):
        """Remove empty directories"""
        for root, dirs, files in os.walk(self.base_dir, topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                try:
                    os.rmdir(dir_path)
                    logger.info(f"Removed empty directory: {dir_path}")
                except OSError:
                    # Directory not empty
                    pass

    def update_metadata(self):
        """Update metadata.json with current dataset structure"""
        metadata = {
            "dataset_info": {
                "name": "Product Verification Dataset",
                "version": "1.0",
                "description": "Dataset for training product verification models",
                "categories": self.categories,
                "total_categories": sum(len(subcats) for subcats in self.categories.values()),
                "total_brands": sum(len(brands) for subcats in self.categories.values() for brands in subcats.values()),
                "image_types": ["original", "fake"]
            }
        }

        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Updated metadata file: {self.metadata_path}")

    def reorganize(self):
        """Run the complete reorganization process"""
        logger.info("Starting dataset reorganization...")
        self.create_directory_structure()
        self.move_existing_files()
        self.cleanup_empty_directories()
        self.update_metadata()
        logger.info("Dataset reorganization completed!")

if __name__ == "__main__":
    reorganizer = DatasetReorganizer()
    reorganizer.reorganize() 