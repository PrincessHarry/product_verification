import os
import requests
from PIL import Image
import io
import time
from typing import List, Dict
import json

class DatasetImageDownloader:
    def __init__(self):
        self.base_dir = "dataset/images"
        self.products = {
            "Nike": {
                "Air_Max": {
                    "original": [
                        "https://static.nike.com/a/images/c_limit,w_592,f_auto/t_product_v1/e6da41fa-1be4-4ce5-b89c-22be4f1f02d4/air-max-270-mens-shoes-KkLcGR.png",
                        "https://static.nike.com/a/images/c_limit,w_592,f_auto/t_product_v1/00b457c1-b884-4c21-8544-a4a95654a8f1/air-max-270-mens-shoes-KkLcGR.png",
                        "https://static.nike.com/a/images/c_limit,w_592,f_auto/t_product_v1/a42a5d53-2f99-4e78-a081-9d07a2d0774a/air-max-270-mens-shoes-KkLcGR.png"
                    ],
                    "fake": [
                        "https://i.ebayimg.com/images/g/K~QAAOSwH4VjYvs1/s-l1600.jpg",
                        "https://i.ebayimg.com/images/g/mhwAAOSwxH1jYvs2/s-l1600.jpg"
                    ]
                }
            },
            "iPhone": {
                "13": {
                    "original": [
                        "https://store.storeimages.cdn-apple.com/4668/as-images.apple.com/is/iphone-13-blue-select-2021?wid=470&hei=556&fmt=jpeg&qlt=95&.v=1645572386470",
                        "https://store.storeimages.cdn-apple.com/4668/as-images.apple.com/is/iphone-13-midnight-select-2021?wid=470&hei=556&fmt=jpeg&qlt=95&.v=1645572386470",
                        "https://store.storeimages.cdn-apple.com/4668/as-images.apple.com/is/iphone-13-pink-select-2021?wid=470&hei=556&fmt=jpeg&qlt=95&.v=1645572386470"
                    ],
                    "fake": [
                        "https://i.ebayimg.com/images/g/qVIAAOSwKjJk~UiE/s-l1600.jpg",
                        "https://i.ebayimg.com/images/g/V~sAAOSwKYdk~UiF/s-l1600.jpg"
                    ]
                }
            },
            "Rolex": {
                "Submariner": {
                    "original": [
                        "https://www.swissluxury.com/product_images/116610LN.jpg",
                        "https://www.swissluxury.com/product_images/116619LB.jpg",
                        "https://www.swissluxury.com/product_images/116613LN.jpg"
                    ],
                    "fake": [
                        "https://i.ebayimg.com/images/g/q4QAAOSwPgxk0xUh/s-l1600.jpg",
                        "https://i.ebayimg.com/images/g/q4QAAOSwPgxk0xUh/s-l1600.jpg"
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
        for brand, categories in self.products.items():
            for category, types in categories.items():
                for img_type, urls in types.items():
                    for i, url in enumerate(urls, 1):
                        save_dir = os.path.join(self.base_dir, brand, category, img_type)
                        save_path = os.path.join(save_dir, f"{img_type}_{i}.jpg")
                        
                        if not os.path.exists(save_path):
                            if self.download_image(url, save_path):
                                time.sleep(1)  # Be nice to servers

if __name__ == "__main__":
    downloader = DatasetImageDownloader()
    downloader.download_all_images() 