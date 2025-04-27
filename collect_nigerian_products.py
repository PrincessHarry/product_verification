#!/usr/bin/env python
"""
Script to collect data for Nigerian products.
"""

import os
import requests
from bs4 import BeautifulSoup
from PIL import Image
import json
import logging
import argparse
from io import BytesIO
import time
import re
from urllib.parse import urljoin

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NigerianProductCollector:
    def __init__(self):
        self.base_dir = "dataset/images"
        self.metadata_file = "dataset/metadata.json"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.products = {
            "Bama_Mayonnaise": {
                "url": "https://www.merciafoods.ng/product/bama-mayonnaise/",
                "category": "Food",
                "subcategory": "Condiments"
            },
            "Peak_Milk": {
                "url": "https://www.peakmilk.com.ng/assortment/powder/peak-full-cream-instant-milk-powder-sachet/",
                "category": "Food",
                "subcategory": "Dairy"
            }
        }
    
    def clean_dataset(self):
        """Remove existing data from dataset directory"""
        if os.path.exists(self.base_dir):
            for root, dirs, files in os.walk(self.base_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            logger.info("Cleaned existing dataset")
        
        if os.path.exists(self.metadata_file):
            os.remove(self.metadata_file)
            logger.info("Removed existing metadata file")
    
    def download_image(self, url, save_path):
        """Download image from URL and save to path"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # Open and convert image
            img = Image.open(BytesIO(response.content))
            
            # Convert RGBA to RGB if necessary
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
                img = background
            
            # Save as JPEG
            img.save(save_path, 'JPEG', quality=95)
            logger.info(f"Downloaded image to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading image from {url}: {str(e)}")
            return False
    
    def extract_images_from_page(self, url):
        """Extract image URLs from a webpage"""
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all image tags
            images = []
            for img in soup.find_all('img'):
                src = img.get('src', '')
                if src:
                    # Convert relative URLs to absolute
                    if src.startswith('/'):
                        src = urljoin(url, src)
                    # Skip small icons and logos
                    if not any(x in src.lower() for x in ['icon', 'logo', 'payment', 'social']):
                        images.append(src)
            
            # Also look for images in CSS background
            for elem in soup.find_all(style=True):
                style = elem['style']
                urls = re.findall(r'url\(["\']?(.*?)["\']?\)', style)
                for url in urls:
                    if not any(x in url.lower() for x in ['icon', 'logo', 'payment', 'social']):
                        images.append(url)
            
            return list(set(images))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error extracting images from {url}: {str(e)}")
            return []
    
    def collect_data(self):
        """Collect data for all products"""
        metadata = {}
        
        for product_name, product_info in self.products.items():
            category = product_info['category']
            subcategory = product_info['subcategory']
            url = product_info['url']
            
            # Create directory structure
            product_dir = os.path.join(self.base_dir, category, subcategory, product_name)
            original_dir = os.path.join(product_dir, "original")
            fake_dir = os.path.join(product_dir, "fake")
            
            os.makedirs(original_dir, exist_ok=True)
            os.makedirs(fake_dir, exist_ok=True)
            
            # Initialize metadata for this product
            metadata[product_name] = {
                "category": category,
                "subcategory": subcategory,
                "original": [],
                "fake": []
            }
            
            # Extract and download original images
            logger.info(f"Collecting original images for {product_name}")
            image_urls = self.extract_images_from_page(url)
            
            for i, img_url in enumerate(image_urls, 1):
                if img_url.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    filename = f"original_{i}.jpg"
                    save_path = os.path.join(original_dir, filename)
                    
                    if self.download_image(img_url, save_path):
                        metadata[product_name]["original"].append({
                            "filename": filename,
                            "path": os.path.relpath(save_path, self.base_dir),
                            "source_url": img_url
                        })
                    
                    time.sleep(1)  # Be nice to servers
        
        return metadata
    
    def update_metadata(self, metadata):
        """Update metadata file with collected data information"""
        os.makedirs(os.path.dirname(self.metadata_file), exist_ok=True)
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logger.info(f"Updated metadata file: {self.metadata_file}")

def main():
    parser = argparse.ArgumentParser(description='Collect Nigerian product data')
    parser.add_argument('--clean', action='store_true', help='Clean existing dataset before collecting')
    
    args = parser.parse_args()
    
    collector = NigerianProductCollector()
    
    if args.clean:
        collector.clean_dataset()
    
    metadata = collector.collect_data()
    collector.update_metadata(metadata)
    
    logger.info("Data collection completed!")

if __name__ == "__main__":
    main() 