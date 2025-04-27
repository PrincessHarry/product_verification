import os
import requests
import json
import logging
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import time
import random
from fake_useragent import UserAgent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CreamCollector:
    def __init__(self):
        self.base_dir = "dataset/images/Cosmetics/Cream"
        self.headers = {'User-Agent': UserAgent().random}
        self.create_base_directories()

    def create_base_directories(self):
        """Create the base directory structure for cream products."""
        brands = [
            # International brands
            "La Mer", "La Prairie", "Tatcha", "Drunk Elephant",
            # Nigerian and African brands
            "Fair & White", "Pure White", "Bio Claire", "Caro White",
            "QEI+", "Perfect White", "Rapid White", "Skin Light",
            "Movate", "Crusader", "Tura", "Venus",
            "Clear Essence", "Fashion Fair", "Makari"
        ]

        for brand in brands:
            for img_type in ['original', 'fake']:
                path = os.path.join(self.base_dir, brand.replace(' ', '_'), img_type)
                os.makedirs(path, exist_ok=True)
                logger.info(f"Created directory: {path}")

    def download_image(self, url, save_path):
        """Download and save an image from a URL."""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            
            # Convert RGBA to RGB if necessary
            if img.mode == 'RGBA':
                img = img.convert('RGB')
                
            img.save(save_path, 'JPEG', quality=95)
            logger.info(f"Successfully downloaded: {save_path}")
            return True
        except Exception as e:
            logger.error(f"Error downloading {url}: {str(e)}")
            return False

    def collect_product_images(self, brand):
        """Collect images for a specific cream brand."""
        brand_dir = os.path.join(self.base_dir, brand.replace(' ', '_'))
        
        # Define image sources based on brand type
        sources = {
            'original': {
                'international': [
                    f"https://www.sephora.com/product/{brand.lower().replace(' ', '-')}",
                    f"https://www.nordstrom.com/s/{brand.lower().replace(' ', '-')}",
                    f"https://www.bloomingdales.com/shop/{brand.lower().replace(' ', '-')}"
                ],
                'nigerian': [
                    f"https://www.jumia.com.ng/catalog/?q={brand.replace(' ', '+')}",
                    f"https://www.konga.com/search?search={brand.replace(' ', '+')}"
                ]
            },
            'fake': [
                f"https://www.google.com/search?q=counterfeit+{brand.replace(' ', '+')}+cream&tbm=isch",
                f"https://www.bing.com/images/search?q=fake+{brand.replace(' ', '+')}+cream"
            ]
        }

        # Determine if it's an international or Nigerian brand
        is_international = brand in ["La Mer", "La Prairie", "Tatcha", "Drunk Elephant"]
        
        for img_type in ['original', 'fake']:
            save_dir = os.path.join(brand_dir, img_type)
            
            if img_type == 'original':
                source_list = sources['original']['international' if is_international else 'nigerian']
            else:
                source_list = sources['fake']

            for source_url in source_list:
                try:
                    # Implement source-specific scraping logic here
                    image_urls = self._scrape_image_urls(source_url)
                    
                    for idx, url in enumerate(image_urls[:5], 1):  # Limit to 5 images per source
                        save_path = os.path.join(save_dir, f"{img_type}_{idx}.jpg")
                        if not os.path.exists(save_path):
                            if self.download_image(url, save_path):
                                time.sleep(random.uniform(1, 3))  # Be nice to servers
                except Exception as e:
                    logger.error(f"Error processing {source_url}: {str(e)}")

    def _scrape_image_urls(self, url: str) -> list:
        """Scrape image URLs from a webpage."""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find image elements
            images = []
            for img in soup.find_all('img'):
                src = img.get('src') or img.get('data-src')
                if src and any(ext in src.lower() for ext in ['.jpg', '.jpeg', '.png']):
                    images.append(src)
            
            return images
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return []

    def collect_all_data(self):
        """Collect data for all cream products."""
        brands = [
            # International brands
            "La Mer", "La Prairie", "Tatcha", "Drunk Elephant",
            # Nigerian and African brands
            "Fair & White", "Pure White", "Bio Claire", "Caro White",
            "QEI+", "Perfect White", "Rapid White", "Skin Light",
            "Movate", "Crusader", "Tura", "Venus",
            "Clear Essence", "Fashion Fair", "Makari"
        ]

        for brand in brands:
            logger.info(f"Collecting images for: {brand}")
            self.collect_product_images(brand)
            time.sleep(random.uniform(2, 5))  # Random delay between brands

if __name__ == "__main__":
    collector = CreamCollector()
    collector.collect_all_data() 