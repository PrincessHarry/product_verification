import os
import shutil
from typing import Dict, List
import json

class DatasetOrganizer:
    def __init__(self):
        self.base_dir = "dataset/images"
        self.categories = {
            "Electronics": {
                "Smartphones": ["iPhone", "Samsung", "Xiaomi"],
                "Laptops": ["Apple", "Dell", "HP"],
                "Watches": ["Apple", "Samsung", "Rolex"],
                "Accessories": ["AirPods", "Galaxy Buds", "Smart Watches"]
            },
            "Fashion": {
                "Shoes": ["Nike", "Adidas", "Under Armour"],
                "Bags": ["Louis Vuitton", "Gucci", "Prada"],
                "Accessories": ["Rolex", "Cartier", "Hermes"],
                "Watches": ["Rolex", "Omega", "Tag Heuer"]
            },
            "Cosmetics": {
                "Skincare": ["La Mer", "SK-II", "Estee Lauder", "Clinique"],
                "Makeup": ["MAC", "Dior", "Chanel", "NARS"],
                "Haircare": ["Kerastase", "Olaplex", "Living Proof"],
                "Bodycare": ["L'Occitane", "Jo Malone", "Molton Brown"],
                "Soap": ["Dove", "L'Occitane", "Aesop", "Fresh"],
                "Cream": [
                    "La Mer", "La Prairie", "Tatcha", "Drunk Elephant",
                    "Fair & White", "Pure White", "Bio Claire", "Caro White",
                    "QEI+", "Perfect White", "Rapid White", "Skin Light",
                    "Movate", "Crusader", "Tura", "Venus",
                    "Clear Essence", "Fashion Fair", "Makari"
                ]
            },
            "Beauty": {
                "Skincare": ["Estee Lauder", "La Mer", "SK-II"],
                "Makeup": ["MAC", "Dior", "Chanel"],
                "Fragrance": ["Chanel", "Dior", "Gucci"],
                "Fragrances": ["Chanel", "Tom Ford", "Jo Malone"]
            },
            "Food": {
                "Snacks": ["Pringles", "Lay's", "Doritos"],
                "Beverages": ["Coca-Cola", "Pepsi", "Nestle"],
                "Condiments": ["Heinz", "Kraft", "McCormick"],
                "Dairy": ["Danone", "Yakult", "Activia"]
            },
            "Pharmaceuticals": {
                "Prescription": ["Pfizer", "Novartis", "Merck"],
                "OTC": ["Johnson & Johnson", "Bayer", "GSK"],
                "Supplements": ["Nature Made", "Centrum", "One A Day"],
                "Pain_Relief": ["Advil", "Tylenol", "Aleve"],
                "Antibiotics": ["Amoxicillin", "Cipro", "Zithromax"]
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
                        print(f"Created directory: {path}")

    def move_existing_images(self):
        """Move existing images to their new locations"""
        # Map of old directories to new locations
        directory_mapping = {
            "Nike/Air_Max": "Fashion/Shoes/Nike",
            "iPhone/13": "Electronics/Smartphones/iPhone",
            "Rolex/Submariner": "Fashion/Accessories/Rolex",
            # Add more mappings as needed
        }

        for old_path, new_path in directory_mapping.items():
            old_full_path = os.path.join(self.base_dir, old_path)
            new_full_path = os.path.join(self.base_dir, new_path)

            if os.path.exists(old_full_path):
                # Move original images
                old_original = os.path.join(old_full_path, "original")
                new_original = os.path.join(new_full_path, "original")
                if os.path.exists(old_original):
                    os.makedirs(new_original, exist_ok=True)
                    for img in os.listdir(old_original):
                        shutil.move(
                            os.path.join(old_original, img),
                            os.path.join(new_original, img)
                        )
                    print(f"Moved original images from {old_path} to {new_path}")

                # Move fake images
                old_fake = os.path.join(old_full_path, "fake")
                new_fake = os.path.join(new_full_path, "fake")
                if os.path.exists(old_fake):
                    os.makedirs(new_fake, exist_ok=True)
                    for img in os.listdir(old_fake):
                        shutil.move(
                            os.path.join(old_fake, img),
                            os.path.join(new_fake, img)
                        )
                    print(f"Moved fake images from {old_path} to {new_path}")

    def cleanup_empty_directories(self):
        """Remove empty directories"""
        for root, dirs, files in os.walk(self.base_dir, topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                try:
                    os.rmdir(dir_path)
                    print(f"Removed empty directory: {dir_path}")
                except OSError:
                    # Directory not empty
                    pass

    def create_metadata_file(self):
        """Create a metadata.json file with dataset information"""
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

        metadata_path = os.path.join("dataset", "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Created metadata file: {metadata_path}")

    def organize_dataset(self):
        """Run the complete organization process"""
        print("Starting dataset organization...")
        self.create_directory_structure()
        self.move_existing_images()
        self.cleanup_empty_directories()
        self.create_metadata_file()
        print("Dataset organization completed!")

if __name__ == "__main__":
    organizer = DatasetOrganizer()
    organizer.organize_dataset() 