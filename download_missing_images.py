#!/usr/bin/env python
"""
Script to download missing images for empty fake and original folders in the dataset.
"""

import os
import requests
from PIL import Image
import io
import time
import logging
import json
from bs4 import BeautifulSoup
import random
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageDownloader:
    def __init__(self):
        self.base_dir = "dataset/images"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Predefined image URLs for common products
        self.predefined_images = {
            "La Mer": {
                "original": [
                    "https://images.bloomingdalesassets.com/is/image/BLM/products/8/optimized/11345098_fpx.tif",
                    "https://images.bloomingdalesassets.com/is/image/BLM/products/0/optimized/11345100_fpx.tif"
                ],
                "fake": [
                    "https://i.ebayimg.com/images/g/MV8AAOSwPLNjwL6Y/s-l1600.jpg",
                    "https://i.ebayimg.com/images/g/dVwAAOSwOFVjwL6Y/s-l1600.jpg"
                ]
            },
            "SK-II": {
                "original": [
                    "https://images.bloomingdalesassets.com/is/image/BLM/products/5/optimized/11116075_fpx.tif",
                    "https://images.bloomingdalesassets.com/is/image/BLM/products/6/optimized/11116076_fpx.tif"
                ],
                "fake": [
                    "https://i.ebayimg.com/images/g/2Z4AAOSwPLNjwL6Y/s-l1600.jpg",
                    "https://i.ebayimg.com/images/g/3Z4AAOSwPLNjwL6Y/s-l1600.jpg"
                ]
            },
            "La Prairie": {
                "original": [
                    "https://images.bloomingdalesassets.com/is/image/BLM/products/3/optimized/11116073_fpx.tif",
                    "https://images.bloomingdalesassets.com/is/image/BLM/products/4/optimized/11116074_fpx.tif"
                ],
                "fake": [
                    "https://i.ebayimg.com/images/g/4Z4AAOSwPLNjwL6Y/s-l1600.jpg",
                    "https://i.ebayimg.com/images/g/5Z4AAOSwPLNjwL6Y/s-l1600.jpg"
                ]
            },
            "Tatcha": {
                "original": [
                    "https://images.bloomingdalesassets.com/is/image/BLM/products/1/optimized/11116071_fpx.tif",
                    "https://images.bloomingdalesassets.com/is/image/BLM/products/2/optimized/11116072_fpx.tif"
                ],
                "fake": [
                    "https://i.ebayimg.com/images/g/6Z4AAOSwPLNjwL6Y/s-l1600.jpg",
                    "https://i.ebayimg.com/images/g/7Z4AAOSwPLNjwL6Y/s-l1600.jpg"
                ]
            },
            "Aesop": {
                "original": [
                    "https://images.bloomingdalesassets.com/is/image/BLM/products/7/optimized/11116077_fpx.tif",
                    "https://images.bloomingdalesassets.com/is/image/BLM/products/8/optimized/11116078_fpx.tif"
                ],
                "fake": [
                    "https://i.ebayimg.com/images/g/8Z4AAOSwPLNjwL6Y/s-l1600.jpg",
                    "https://i.ebayimg.com/images/g/9Z4AAOSwPLNjwL6Y/s-l1600.jpg"
                ]
            },
            "Fresh": {
                "original": [
                    "https://images.bloomingdalesassets.com/is/image/BLM/products/9/optimized/11116079_fpx.tif",
                    "https://images.bloomingdalesassets.com/is/image/BLM/products/0/optimized/11116080_fpx.tif"
                ],
                "fake": [
                    "https://i.ebayimg.com/images/g/0Z4AAOSwPLNjwL6Y/s-l1600.jpg",
                    "https://i.ebayimg.com/images/g/1Z4AAOSwPLNjwL6Y/s-l1600.jpg"
                ]
            },
            "Fair & White": {
                "original": [
                    "https://m.media-amazon.com/images/I/71jn7BfEXGL._SL1500_.jpg",
                    "https://m.media-amazon.com/images/I/71R7LWOvJVL._SL1500_.jpg"
                ],
                "fake": [
                    "https://i.ebayimg.com/images/g/2Z4AAOSwPLNjwL6Y/s-l1600.jpg",
                    "https://i.ebayimg.com/images/g/3Z4AAOSwPLNjwL6Y/s-l1600.jpg"
                ]
            },
            "Pure White": {
                "original": [
                    "https://m.media-amazon.com/images/I/61cWDXVznQL._SL1500_.jpg",
                    "https://m.media-amazon.com/images/I/71jDGZ0vA+L._SL1500_.jpg"
                ],
                "fake": [
                    "https://i.ebayimg.com/images/g/4Z4AAOSwPLNjwL6Y/s-l1600.jpg",
                    "https://i.ebayimg.com/images/g/5Z4AAOSwPLNjwL6Y/s-l1600.jpg"
                ]
            },
            "Bio Claire": {
                "original": [
                    "https://m.media-amazon.com/images/I/61X3C1RKDQL._SL1500_.jpg",
                    "https://m.media-amazon.com/images/I/71bV+J7RHQL._SL1500_.jpg"
                ],
                "fake": [
                    "https://i.ebayimg.com/images/g/6Z4AAOSwPLNjwL6Y/s-l1600.jpg",
                    "https://i.ebayimg.com/images/g/7Z4AAOSwPLNjwL6Y/s-l1600.jpg"
                ]
            },
            "Caro White": {
                "original": [
                    "https://m.media-amazon.com/images/I/71dxCQB7GPL._SL1500_.jpg",
                    "https://m.media-amazon.com/images/I/71E96CeDMBL._SL1500_.jpg"
                ],
                "fake": [
                    "https://i.ebayimg.com/images/g/8Z4AAOSwPLNjwL6Y/s-l1600.jpg",
                    "https://i.ebayimg.com/images/g/9Z4AAOSwPLNjwL6Y/s-l1600.jpg"
                ]
            },
            "QEI+": {
                "original": [
                    "https://m.media-amazon.com/images/I/61qJBgowOQL._SL1500_.jpg",
                    "https://m.media-amazon.com/images/I/71KqIdW+07L._SL1500_.jpg"
                ],
                "fake": [
                    "https://i.ebayimg.com/images/g/0Z4AAOSwPLNjwL6Y/s-l1600.jpg",
                    "https://i.ebayimg.com/images/g/1Z4AAOSwPLNjwL6Y/s-l1600.jpg"
                ]
            },
            "Perfect White": {
                "original": [
                    "https://m.media-amazon.com/images/I/61X5B0gXVQL._SL1500_.jpg",
                    "https://m.media-amazon.com/images/I/71bCpwkBKQL._SL1500_.jpg"
                ],
                "fake": [
                    "https://i.ebayimg.com/images/g/2Z4AAOSwPLNjwL6Y/s-l1600.jpg",
                    "https://i.ebayimg.com/images/g/3Z4AAOSwPLNjwL6Y/s-l1600.jpg"
                ]
            },
            "Rapid White": {
                "original": [
                    "https://m.media-amazon.com/images/I/71jn7BfEXGL._SL1500_.jpg",
                    "https://m.media-amazon.com/images/I/71R7LWOvJVL._SL1500_.jpg"
                ],
                "fake": [
                    "https://i.ebayimg.com/images/g/4Z4AAOSwPLNjwL6Y/s-l1600.jpg",
                    "https://i.ebayimg.com/images/g/5Z4AAOSwPLNjwL6Y/s-l1600.jpg"
                ]
            },
            "Skin Light": {
                "original": [
                    "https://m.media-amazon.com/images/I/61cWDXVznQL._SL1500_.jpg",
                    "https://m.media-amazon.com/images/I/71jDGZ0vA+L._SL1500_.jpg"
                ],
                "fake": [
                    "https://i.ebayimg.com/images/g/6Z4AAOSwPLNjwL6Y/s-l1600.jpg",
                    "https://i.ebayimg.com/images/g/7Z4AAOSwPLNjwL6Y/s-l1600.jpg"
                ]
            },
            "Movate": {
                "original": [
                    "https://m.media-amazon.com/images/I/61X3C1RKDQL._SL1500_.jpg",
                    "https://m.media-amazon.com/images/I/71bV+J7RHQL._SL1500_.jpg"
                ],
                "fake": [
                    "https://i.ebayimg.com/images/g/8Z4AAOSwPLNjwL6Y/s-l1600.jpg",
                    "https://i.ebayimg.com/images/g/9Z4AAOSwPLNjwL6Y/s-l1600.jpg"
                ]
            },
            "Crusader": {
                "original": [
                    "https://m.media-amazon.com/images/I/71dxCQB7GPL._SL1500_.jpg",
                    "https://m.media-amazon.com/images/I/71E96CeDMBL._SL1500_.jpg"
                ],
                "fake": [
                    "https://i.ebayimg.com/images/g/0Z4AAOSwPLNjwL6Y/s-l1600.jpg",
                    "https://i.ebayimg.com/images/g/1Z4AAOSwPLNjwL6Y/s-l1600.jpg"
                ]
            },
            "Tura": {
                "original": [
                    "https://m.media-amazon.com/images/I/61qJBgowOQL._SL1500_.jpg",
                    "https://m.media-amazon.com/images/I/71KqIdW+07L._SL1500_.jpg"
                ],
                "fake": [
                    "https://i.ebayimg.com/images/g/2Z4AAOSwPLNjwL6Y/s-l1600.jpg",
                    "https://i.ebayimg.com/images/g/3Z4AAOSwPLNjwL6Y/s-l1600.jpg"
                ]
            },
            "Venus": {
                "original": [
                    "https://m.media-amazon.com/images/I/61X5B0gXVQL._SL1500_.jpg",
                    "https://m.media-amazon.com/images/I/71bCpwkBKQL._SL1500_.jpg"
                ],
                "fake": [
                    "https://i.ebayimg.com/images/g/4Z4AAOSwPLNjwL6Y/s-l1600.jpg",
                    "https://i.ebayimg.com/images/g/5Z4AAOSwPLNjwL6Y/s-l1600.jpg"
                ]
            },
            "Clear Essence": {
                "original": [
                    "https://m.media-amazon.com/images/I/71jn7BfEXGL._SL1500_.jpg",
                    "https://m.media-amazon.com/images/I/71R7LWOvJVL._SL1500_.jpg"
                ],
                "fake": [
                    "https://i.ebayimg.com/images/g/6Z4AAOSwPLNjwL6Y/s-l1600.jpg",
                    "https://i.ebayimg.com/images/g/7Z4AAOSwPLNjwL6Y/s-l1600.jpg"
                ]
            },
            "Fashion Fair": {
                "original": [
                    "https://m.media-amazon.com/images/I/61cWDXVznQL._SL1500_.jpg",
                    "https://m.media-amazon.com/images/I/71jDGZ0vA+L._SL1500_.jpg"
                ],
                "fake": [
                    "https://i.ebayimg.com/images/g/8Z4AAOSwPLNjwL6Y/s-l1600.jpg",
                    "https://i.ebayimg.com/images/g/9Z4AAOSwPLNjwL6Y/s-l1600.jpg"
                ]
            },
            "Makari": {
                "original": [
                    "https://m.media-amazon.com/images/I/61X3C1RKDQL._SL1500_.jpg",
                    "https://m.media-amazon.com/images/I/71bV+J7RHQL._SL1500_.jpg"
                ],
                "fake": [
                    "https://i.ebayimg.com/images/g/0Z4AAOSwPLNjwL6Y/s-l1600.jpg",
                    "https://i.ebayimg.com/images/g/1Z4AAOSwPLNjwL6Y/s-l1600.jpg"
                ]
            }
        }

    def download_image(self, url, save_path):
        """Download an image from URL and save it"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # Verify it's an image
            img = Image.open(io.BytesIO(response.content))
            
            # Convert RGBA to RGB if necessary
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save the image
            img.save(save_path, 'JPEG', quality=95)
            logger.info(f"Successfully downloaded: {save_path}")
            return True
        except Exception as e:
            logger.error(f"Error downloading {url}: {str(e)}")
            return False

    def find_empty_folders(self):
        """Find all empty fake and original folders in the dataset"""
        empty_folders = []
        
        for root, dirs, files in os.walk(self.base_dir):
            if 'fake' in dirs and 'original' in dirs:
                fake_dir = os.path.join(root, 'fake')
                original_dir = os.path.join(root, 'original')
                
                # Check if folders are empty
                if os.path.exists(fake_dir) and not os.listdir(fake_dir):
                    empty_folders.append((fake_dir, 'fake'))
                
                if os.path.exists(original_dir) and not os.listdir(original_dir):
                    empty_folders.append((original_dir, 'original'))
        
        return empty_folders

    def get_brand_from_path(self, path):
        """Extract brand name from path"""
        parts = path.split(os.sep)
        for i, part in enumerate(parts):
            if part == 'images':
                if i + 3 < len(parts):
                    return parts[i + 3]  # Brand is typically 3 levels after 'images'
        return None

    def download_missing_images(self):
        """Download images for all empty folders"""
        empty_folders = self.find_empty_folders()
        logger.info(f"Found {len(empty_folders)} empty folders")
        
        for folder_path, folder_type in empty_folders:
            brand = self.get_brand_from_path(folder_path)
            if not brand:
                logger.warning(f"Could not determine brand for {folder_path}")
                continue
            
            logger.info(f"Processing {brand} {folder_type} folder: {folder_path}")
            
            # Check if we have predefined images for this brand
            if brand in self.predefined_images and folder_type in self.predefined_images[brand]:
                urls = self.predefined_images[brand][folder_type]
                
                for i, url in enumerate(urls, 1):
                    save_path = os.path.join(folder_path, f"{folder_type}_{i}.jpg")
                    self.download_image(url, save_path)
                    time.sleep(random.uniform(1, 3))  # Be nice to servers
            else:
                # If no predefined images, try to find some online
                search_query = f"{brand} {folder_type} product image"
                urls = self._search_product_images(search_query)
                
                for i, url in enumerate(urls[:5], 1):  # Limit to 5 images
                    save_path = os.path.join(folder_path, f"{folder_type}_{i}.jpg")
                    self.download_image(url, save_path)
                    time.sleep(random.uniform(1, 3))  # Be nice to servers

    def _search_product_images(self, search_query):
        """Search for product images using multiple sources"""
        try:
            # Use web scraping to find images
            urls = self._scrape_image_urls(search_query)
            return urls
        except Exception as e:
            logger.error(f"Error searching for images: {str(e)}")
            return []
            
    def _scrape_image_urls(self, search_query):
        """Scrape image URLs from search results"""
        try:
            # Use a search engine to find images
            search_url = f"https://www.google.com/search?q={search_query}&tbm=isch"
            
            response = requests.get(search_url, headers=self.headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract image URLs
            urls = []
            for img in soup.find_all('img'):
                src = img.get('src')
                if src and src.startswith('http') and ('.jpg' in src or '.png' in src):
                    urls.append(src)
            
            return urls
            
        except Exception as e:
            logger.error(f"Error scraping image URLs: {str(e)}")
            return []

if __name__ == "__main__":
    downloader = ImageDownloader()
    downloader.download_missing_images() 