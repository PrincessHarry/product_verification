#!/usr/bin/env python
"""
Unified script for collecting product verification data.
Handles all categories: Cosmetics, Electronics, Fashion, etc.
"""

import os
import requests
from PIL import Image
from io import BytesIO
import time
import json
import logging
from typing import Dict, List
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self):
        self.base_dir = "dataset/images"
        self.metadata_path = "dataset/metadata.json"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Product categories and their data
        self.categories = {
            "Cosmetics": {
                "Cream": {
                    "brands": {
                        "Fair_&_White": {
                            "original": [
                                "https://images.bloomingdalesassets.com/is/image/BLM/products/8/optimized/11345098_fpx.tif",
                                "https://images.bloomingdalesassets.com/is/image/BLM/products/0/optimized/11345100_fpx.tif"
                            ],
                            "fake": [
                                "https://i.ebayimg.com/images/g/MV8AAOSwPLNjwL6Y/s-l1600.jpg",
                                "https://i.ebayimg.com/images/g/dVwAAOSwOFVjwL6Y/s-l1600.jpg"
                            ]
                        },
                        "Pure_White": {
                            "original": [
                                "https://images.bloomingdalesassets.com/is/image/BLM/products/5/optimized/11116075_fpx.tif",
                                "https://images.bloomingdalesassets.com/is/image/BLM/products/6/optimized/11116076_fpx.tif"
                            ],
                            "fake": [
                                "https://i.ebayimg.com/images/g/2Z4AAOSwPLNjwL6Y/s-l1600.jpg",
                                "https://i.ebayimg.com/images/g/3Z4AAOSwPLNjwL6Y/s-l1600.jpg"
                            ]
                        }
                        # Add more brands as needed
                    }
                },
                "Skincare": {
                    "brands": {
                        "La_Mer": {
                            "original": [
                                "https://images.bloomingdalesassets.com/is/image/BLM/products/8/optimized/11345098_fpx.tif",
                                "https://images.bloomingdalesassets.com/is/image/BLM/products/0/optimized/11345100_fpx.tif"
                            ],
                            "fake": [
                                "https://i.ebayimg.com/images/g/MV8AAOSwPLNjwL6Y/s-l1600.jpg",
                                "https://i.ebayimg.com/images/g/dVwAAOSwOFVjwL6Y/s-l1600.jpg"
                            ]
                        }
                        # Add more brands as needed
                    }
                }
            },
            "Electronics": {
                "Smartphones": {
                    "brands": {
                        "iPhone": {
                            "original": [
                                "https://store.storeimages.cdn-apple.com/4982/as-images.apple.com/is/iphone-13-blue-select-2021",
                                "https://store.storeimages.cdn-apple.com/4982/as-images.apple.com/is/iphone-13-mini-blue-select-2021"
                            ],
                            "fake": [
                                "https://i.ebayimg.com/images/g/0~AAAOSw~1Fj~y~Y/s-l1600.jpg",
                                "https://i.ebayimg.com/images/g/0~AAAOSw~1Fj~y~Y/s-l1600.jpg"
                            ]
                        }
                        # Add more brands as needed
                    }
                }
            }
            # Add more categories as needed
        }

    def download_image(self, url: str, save_path: str) -> bool:
        """Download an image from URL and save it"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
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

    def collect_category_data(self, category: str, subcategory: str, brand: str, 
                            image_type: str, urls: List[str]):
        """Collect images for a specific brand and type"""
        save_dir = os.path.join(self.base_dir, category, subcategory, brand, image_type)
        os.makedirs(save_dir, exist_ok=True)

        for i, url in enumerate(urls, 1):
            save_path = os.path.join(save_dir, f"{image_type}_{i}.jpg")
            if not os.path.exists(save_path):
                self.download_image(url, save_path)
                time.sleep(1)  # Be nice to servers

    def collect_all_data(self):
        """Collect data for all categories"""
        for category, subcategories in self.categories.items():
            for subcategory, data in subcategories.items():
                if "brands" in data:
                    for brand, images in data["brands"].items():
                        logger.info(f"Collecting data for {brand} {subcategory}")
                        
                        # Collect original images
                        if "original" in images:
                            self.collect_category_data(category, subcategory, brand, 
                                                     "original", images["original"])
                        
                        # Collect fake images
                        if "fake" in images:
                            self.collect_category_data(category, subcategory, brand, 
                                                     "fake", images["fake"])
                        
                        time.sleep(2)  # Be nice to servers

    def update_metadata(self):
        """Update metadata.json with collected data information"""
        metadata = {
            "dataset_info": {
                "total_images": 0,
                "categories": {}
            }
        }

        for category in os.listdir(self.base_dir):
            category_path = os.path.join(self.base_dir, category)
            if os.path.isdir(category_path):
                metadata["dataset_info"]["categories"][category] = {}
                
                for subcategory in os.listdir(category_path):
                    subcategory_path = os.path.join(category_path, subcategory)
                    if os.path.isdir(subcategory_path):
                        metadata["dataset_info"]["categories"][category][subcategory] = []
                        
                        for brand in os.listdir(subcategory_path):
                            brand_path = os.path.join(subcategory_path, brand)
                            if os.path.isdir(brand_path):
                                original_path = os.path.join(brand_path, "original")
                                fake_path = os.path.join(brand_path, "fake")
                                
                                brand_info = {
                                    "name": brand,
                                    "original_images": len(os.listdir(original_path)) if os.path.exists(original_path) else 0,
                                    "fake_images": len(os.listdir(fake_path)) if os.path.exists(fake_path) else 0
                                }
                                metadata["dataset_info"]["categories"][category][subcategory].append(brand_info)
                                metadata["dataset_info"]["total_images"] += (
                                    brand_info["original_images"] + brand_info["fake_images"]
                                )

        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info("Metadata updated successfully")

def main():
    collector = DataCollector()
    collector.collect_all_data()
    collector.update_metadata()

if __name__ == "__main__":
    main() 