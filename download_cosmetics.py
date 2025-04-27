import os
import requests
from PIL import Image
import io
import time
from typing import List, Dict
import json

class CosmeticsDataDownloader:
    def __init__(self):
        self.base_dir = "dataset/images/Cosmetics"
        self.products = {
            "Skincare": {
                "La_Mer": {
                    "original": [
                        "https://images.bloomingdalesassets.com/is/image/BLM/products/8/optimized/11345098_fpx.tif",
                        "https://images.bloomingdalesassets.com/is/image/BLM/products/0/optimized/11345100_fpx.tif"
                    ],
                    "fake": [
                        "https://i.ebayimg.com/images/g/MV8AAOSwPLNjwL6Y/s-l1600.jpg",
                        "https://i.ebayimg.com/images/g/dVwAAOSwOFVjwL6Y/s-l1600.jpg"
                    ]
                },
                "SK_II": {
                    "original": [
                        "https://images.bloomingdalesassets.com/is/image/BLM/products/5/optimized/11116075_fpx.tif",
                        "https://images.bloomingdalesassets.com/is/image/BLM/products/6/optimized/11116076_fpx.tif"
                    ],
                    "fake": [
                        "https://i.ebayimg.com/images/g/2Z4AAOSwPLNjwL6Y/s-l1600.jpg",
                        "https://i.ebayimg.com/images/g/3Z4AAOSwPLNjwL6Y/s-l1600.jpg"
                    ]
                }
            },
            "Cream": {
                "La_Prairie": {
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
                }
            },
            "Soap": {
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
                        "https://i.ebayimg.com/images/g/AZ4AAOSwPLNjwL6Y/s-l1600.jpg",
                        "https://i.ebayimg.com/images/g/BZ4AAOSwPLNjwL6Y/s-l1600.jpg"
                    ]
                }
            }
        }

    def download_image(self, url: str, save_path: str) -> bool:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Open the image to verify it's valid
            img = Image.open(io.BytesIO(response.content))
            
            # Convert RGBA to RGB if necessary
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            
            # Save the image
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            img.save(save_path, 'JPEG', quality=95)
            print(f"Successfully downloaded: {save_path}")
            return True
        except Exception as e:
            print(f"Error downloading {url}: {str(e)}")
            return False

    def download_all_images(self):
        for category, brands in self.products.items():
            for brand, types in brands.items():
                for img_type, urls in types.items():
                    for i, url in enumerate(urls, 1):
                        save_dir = os.path.join(self.base_dir, category, brand, img_type)
                        save_path = os.path.join(save_dir, f"{img_type}_{i}.jpg")
                        
                        if not os.path.exists(save_path):
                            if self.download_image(url, save_path):
                                time.sleep(1)  # Be nice to servers

if __name__ == "__main__":
    downloader = CosmeticsDataDownloader()
    downloader.download_all_images() 